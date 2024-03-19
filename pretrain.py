#!/usr/bin/env python3
""" Pre-Train Python Script
Heavily based on the training script provided by timm.

Title: pytorch-image-models
Author: Ross Wightman
Date: 2021
Availability: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
"""

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import shutil
import math
import random as rand

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import resume_checkpoint, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ApexScaler, NativeScaler

import webdataset as wds
import torch.distributed as dist
import pandas as pd

from data.loader import create_loader, create_transform_webdataset
from models.factory import create_model, safe_model_name
from scheduler.scheduler_factory import create_scheduler
from utils.summary import original_update_summary

from models.resnet import ResNet18, ResNet34, ResNet50
from models.vgg import VGG
from models.wideresnet import WideResNet
from torchvision import transforms
from optimizers.lion import Lion
from optimizers.lion_mshift import Lion_mshift
from optimizers.lion_mvote import Lion_MVote

def print0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


parser = argparse.ArgumentParser(description='PreTraining')

# Dataset / Model parameters
parser.add_argument('--train_data_dir', default='/mnt/nfs/datasets/ILSVRC2012/train/',
                    help='path to dataset, do not used when using WebDataSet')
parser.add_argument('--eval_data_dir', default='/mnt/nfs/datasets/ILSVRC2012/val/',
                    help='path to dataset, do not used when using WebDataSet')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: deit_tiny_patch16_224)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', nargs=3, type=int, metavar='N N N', default= [3, 224, 224],
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Web Datasets
parser.add_argument('--trainshards', default=None,
                    help='path/URL for ImageNet shards')
parser.add_argument('-w', '--webdataset', action='store_true', default=False,
                    help='Using webdata to create DataSet from .tar files')
parser.add_argument('--dataset_size', default=None, type=int,
                    help='Number of Images in the dataset, set to num_classes * 1000 if None')

# Optimizer parameters
parser.add_argument('--optimizer_name', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup-iter', type=int, default=0, metavar='N',
                    help='iter to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--repeated-aug', action='store_true',
                    help='Use repeated augmentation')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('--interval-saved-epochs', type=int, default=None,
                    help='the interval epoch to keep checkpoint (other than args.checkpoint_hist)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default=None, type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--project-name', default='YOUR_WANDB_PPOJECT_NAME', type=str,
                    help='set wandb project name')
parser.add_argument('--group-name', default='YOUR_WANDB_GROUP_NAME', type=str,
                    help='set wandb group name')
parser.add_argument('--interval_saved_epochs', default=10, type=int, metavar='EVAL_METRIC',
                    help='interval_saved_epochs')

# shampoo
parser.add_argument('--beta2', default=0.85, type=float)
parser.add_argument('--matrix_eps', default=1.0e-6, type=float)
parser.add_argument('--start_preconditioning_step', default=25, type=int)
parser.add_argument('--preconditioning_compute_steps', default=10, type=int)
parser.add_argument('--statistics_compute_steps', default=10, type=int)
parser.add_argument('--early_phase_ratio', default=0, type=float)
parser.add_argument('--early_preconditioning_compute_steps', default=10, type=int)
parser.add_argument('--early_statistics_compute_steps', default=100, type=int)
parser.add_argument('--shampoo_block_size', default=128, type=int)
parser.add_argument('--gradient_value_clip', default=-1, type=float)
parser.add_argument('--grafting', default='AdaGrad', type=str, choices=['None', 'SGD', 'AdaGrad'])

parser.add_argument('--interval_cosine_thres', default=-1, type=float)
parser.add_argument('--interval_scheduling_factor', default=1, type=float)
parser.add_argument('--inverse_exponent', default=0, type=float)
parser.add_argument('--use_inverse', action='store_true', default=False,
                    help='shampoo_inverse')
parser.add_argument('--dmp_opt', default='mean', type=str)

parser.add_argument('--log_optimizer_state', action='store_true', default=False)
parser.add_argument("--state_interval", type=int, default=1, help="Interval to the previous optimizer state")
parser.add_argument("--log_weight_iters", type=str, default='None', help="Interval to the previous optimizer state")
parser.add_argument('--ckpo_with_current_time', action='store_true', default=False)
parser.add_argument("--momentum_sync_freq", type=int, default=1, help="Frequency for sync momentum")

# CIFAR Dataset
parser.add_argument('--use_cifar', action='store_true', default=False,
                    help='use cifar dataset')

def _parse_args():
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_weight_iters == 'None':
        args.log_weight_iters = []
    else:
        args.log_weight_iters = args.log_weight_iters.split(',')

    args.prefetcher = not args.no_prefetcher
    args.distributed = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1')) > 1
    args.local_rank = 0
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        # initialize torch.distributed using MPI
        master_addr = os.getenv("MASTER_ADDR", default="localhost")
        master_port = os.getenv('MASTER_PORT', default='8888')
        method = "tcp://{}:{}".format(master_addr, master_port)
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))  # global rank
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))

        ngpus_per_node = torch.cuda.device_count()
        node = rank // ngpus_per_node
        args.local_rank = rank % ngpus_per_node
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=method, world_size=world_size, rank=rank)
        args.rank = rank
        args.world_size = world_size
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d:%d, total %d.'
                     % (args.local_rank, node, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.log_wandb and args.rank == 0:
        if has_wandb:
            wandb.init(config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    if args.use_cifar:
        if args.dataset.lower() == 'cifar10':
            args.num_classes = 10
        if args.dataset.lower()== 'cifar100':
            args.num_classes = 100
        args.input_size = [3, 32, 32]
        if args.model == 'resnet18':
            model = ResNet18(num_classes=args.num_classes)
        elif args.model == 'resnet34':
            model = ResNet34(num_classes=args.num_classes)
        elif args.model == 'resnet50':
            model = ResNet50(num_classes=args.num_classes)
        elif args.model == 'wideresnet16':
            model = WideResNet(depth=16, num_classes=args.num_classes, widen_factor=8, dropRate=args.drop)
        elif args.model == 'wideresnet28':
            model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10, dropRate=args.drop)
        elif 'VGG' in args.model:
            model = VGG(vgg_name=args.model, num_classes=args.num_classes)
        else:
            print('No model matched')
    else:
        model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    param_name_dict = {}
    prev_weight_dict = {}
    prev_grad_dict = {}
    prev_momentum_dict = {}
    for name, param in model.named_parameters():
        param_name_dict[param] = name

    if args.rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        # # mkdir output dir
        # os.makedirs(args.output, exist_ok=True)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # setup augmentation batch splits for contrastive loss
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # move model to GPU
    model.cuda()

    # Choose the DataSet Selector
    if args.webdataset:
        print0("\n\n=> Loading DataSet with WebDataset using .tars")
        collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        # Transforms from Torchvision
        # create data loaders w/ augmentation pipeline
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
        transform_train = create_transform_webdataset(
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            repeated_aug=args.repeated_aug
        )

        if args.dataset_size:
            dataset_size = args.dataset_size
        elif "_2ki" in args.trainshards:
            dataset_size = args.num_classes * 2000
        else:
            dataset_size = args.num_classes * 1000

        train_dataset = (
            wds.Dataset(args.trainshards)
                .shuffle(dataset_size)
                .decode("pil")
                .rename(image="jpg;jpeg;JPEG;png", target="cls")
                .map_dict(image=transform_train)
                .to_tuple("image", "target")
        )
        loader_train = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=args.workers)
        train_dataset = train_dataset.batched(args.batch_size, partial=False)

        number_of_batches = dataset_size // (args.batch_size * args.world_size)
        print0("dataset_size:{}, batch_size:{}, world_size(Total Devices):{}".format(dataset_size, args.batch_size,
                                                                                        args.world_size))
        print0("----> Number of batches to be processed per GPU = {}".format(number_of_batches))
        loader_train = loader_train.repeat(2).slice(number_of_batches)
        # This only sets the value returned by the len() function; nothing else uses it,
        # but some frameworks care about it.
        loader_train.length = number_of_batches

    elif args.use_cifar:
        from data.autoaugment import CIFAR10Policy
        from data.cutout import Cutout
        datasetname = args.dataset.lower()
        normalize = transforms.Normalize(   mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                normalize
        ])
        val_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize
        ])

        if datasetname == 'cifar10':
            cutout = Cutout(n_holes=1, length=16)
            train_transform.transforms.append(cutout)
            dataset_train = torchvision.datasets.CIFAR10(root='data/',
                                        train=True,
                                        download=True,
                                        transform = train_transform)
            dataset_eval = torchvision.datasets.CIFAR10(root='data/',
                                        train=False,
                                        download=True,
                                        transform = val_transform)
        elif datasetname == 'cifar100':
            cutout = Cutout(n_holes=1, length=8)
            train_transform.transforms.append(cutout)
            dataset_train = torchvision.datasets.CIFAR100(root='data/',
                                        train=True,
                                        download=True,
                                        transform = train_transform)
            dataset_eval = torchvision.datasets.CIFAR100(root='data/',
                                        train=False,
                                        download=True,
                                        transform = val_transform)
        
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=args.workers)
        loader_eval = torch.utils.data.DataLoader(dataset=dataset_eval,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=args.workers)
        collate_fn = None
        mixup_fn = None
        mixup_active = False

    else:
        dataset_train = create_dataset(
            args.dataset,
            root=args.train_data_dir, split=args.train_split, is_training=True,
            batch_size=args.batch_size, repeats=args.epoch_repeats)
        dataset_eval = create_dataset(
            args.dataset, root=args.eval_data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

        # setup mixup / cutmix
        collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        # wrap dataset in AugMix helper
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            repeated_aug=args.repeated_aug
        )
    
        loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        )
    if args.optimizer_name == 'lion_mshift' and args.momentum_sync_freq ==0:
        args.optimizer_name = 'lion'
    if args.optimizer_name == 'lion_mvote' and args.momentum_sync_freq ==0:
        args.optimizer_name = 'lion'
    if args.optimizer_name == 'lion':
        optimizer = Lion(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer_name == 'lion_mvote':
        optimizer = Lion_MVote(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), weight_decay=args.weight_decay)
        model.require_backward_grad_sync = False
    elif args.optimizer_name == 'lion_mshift':
        optimizer = Lion_mshift(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), weight_decay=args.weight_decay)
        model.require_backward_grad_sync = False
    elif args.optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), weight_decay=args.weight_decay)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.rank == 0)
        if args.rank == 0:
            _logger.info('resume epoch: {}'.format(resume_epoch))

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1

    # setup learning rate schedule and starting epoch
    iter_per_epoch = len(loader_train)
    lr_scheduler, num_iters = create_scheduler(args, optimizer, len(loader_train))
    num_epochs = args.epochs + args.cooldown_epochs
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if 'iter' in args.sched:
            lr_scheduler.step_update(start_epoch * len(loader_train))
        else:
            lr_scheduler.step(start_epoch)

    if args.rank == 0:
        _logger.info('iter per epoch: {}'.format(iter_per_epoch))
        _logger.info('Scheduled iters: {}'.format(num_iters))
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    # if args.rank == 0:
    #     exp_name = '-'.join([
    #         datetime.now().strftime("%Y%m%d-%H%M%S.%f"),
    #         safe_model_name(args.model),
    #         str(data_config['input_size'][-1])
    #     ])
    #     output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    #     decreasing = True if eval_metric == 'loss' else False
    #     saver = CheckpointSaver(
    #         model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
    #         checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    #     with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
    #         f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):

            if args.webdataset is not True:
                if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                    loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, mixup_fn=mixup_fn,
                param_name_dict = param_name_dict, prev_weight_dict=prev_weight_dict,prev_grad_dict=prev_grad_dict,prev_momentum_dict=prev_momentum_dict)
            
            ## EVAL
            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )
            if math.isnan(eval_metrics['loss']):
                break
            ##

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.rank == 0:
                original_update_summary(
                    epoch, train_metrics, eval_metrics,
                    log_wandb=args.log_wandb and has_wandb)

            #if saver is not None:
            #    # save proper checkpoint with eval metric
            #    save_metric = train_metrics[eval_metric]
            #    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            #    if args.interval_saved_epochs is not None and epoch % args.interval_saved_epochs == 0:
            #        # only the last {args.checkpoint_hist} checkpoints will be kept
            #        # this makes checkpoint every args.interval_saved_epochs to be saved
            #        if args.output:
            #            checkpoint_file = f'{args.output}/{args.experiment}/checkpoint-{epoch}.pth.tar'
            #            target_file = f'{args.output}/{args.experiment}/held-checkpoint-{epoch}.pth.tar'
            #        else:
            #            checkpoint_file = f'output/train/{args.experiment}/checkpoint-{epoch}.pth.tar'
            #            target_file = f'output/train/{args.experiment}/held-checkpoint-{epoch}.pth.tar'
            #        if os.path.exists(checkpoint_file):
            #            shutil.copyfile(checkpoint_file, target_file)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, mixup_fn=None,
        param_name_dict = None, prev_weight_dict=None,prev_grad_dict=None,prev_momentum_dict=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    cos_func = torch.nn.CosineSimilarity(dim=0)
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        # if loss_scaler is not None:
        #     loss_scaler(
        #         loss, optimizer,
        #         clip_grad=args.clip_grad, clip_mode=args.clip_mode,
        #         parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
        #         create_graph=second_order)
        if loss_scaler is None:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            if args.optimizer_name == 'lion_mshift' or args.optimizer_name == 'lion_mvote':
                optimizer.step(sync_momentum = (num_updates % args.momentum_sync_freq==0))
            else:
                optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        state_info = None
        if args.log_optimizer_state and args.rank == 0:
            if batch_idx % args.log_interval == 0:
                state_info = {}
                for p in optimizer.state.keys():
                    param_name = param_name_dict[p]
                    weight = p.data
                    grad = p.grad
                    momentum = optimizer.state[p]["exp_avg"]
                    state_info['weight_norm/'+param_name] = torch.norm(p.data).item()
                    state_info['grad_norm/'+param_name] = torch.norm(p.grad).item()
                    state_info['momentum_norm/'+param_name] = torch.norm(momentum).item()
                    state_info['weight_norm_element/'+param_name] = torch.abs(p.data).mean(dtype=torch.float32).item()
                    state_info['grad_norm_element/'+param_name] = torch.abs(p.grad).mean(dtype=torch.float32).item()
                    state_info['momentum_norm_element/'+param_name] = torch.abs(momentum).mean(dtype=torch.float32).item()
                    if param_name in prev_weight_dict.keys():
                        prev_weight = prev_weight_dict[param_name]
                        state_info['weight_relative_error/'+param_name] = torch.norm(weight - prev_weight).item() / state_info['weight_norm/'+param_name]
                        state_info['weight_relative_error_element/'+param_name] = torch.abs(grad - prev_weight).mean(dtype=torch.float32).item() / state_info['weight_norm_element/'+param_name]
                        state_info['weight_cosine_sim/'+param_name] = cos_func(weight.view(-1), prev_weight.view(-1))
                        state_info['weight_norm_ratio/'+param_name] = state_info['weight_norm/'+param_name] / torch.norm(prev_weight).item()
                        state_info['weight_norm_element_ratio/'+param_name] = state_info['weight_norm_element/'+param_name] / torch.abs(prev_weight).mean(dtype=torch.float32).item()
                    if param_name in prev_grad_dict.keys():
                        prev_grad = prev_grad_dict[param_name]
                        state_info['grad_relative_error/'+param_name] = torch.norm(grad - prev_grad).item() / state_info['grad_norm/'+param_name]
                        state_info['grad_relative_error_element/'+param_name] = torch.abs(grad - prev_grad).mean(dtype=torch.float32).item() / state_info['grad_norm_element/'+param_name]
                        state_info['grad_cosine_sim/'+param_name] = cos_func(grad.view(-1), prev_grad.view(-1))
                        state_info['grad_norm_ratio/'+param_name] = state_info['grad_norm/'+param_name] / torch.norm(prev_grad).item()
                        state_info['grad_norm_element_ratio/'+param_name] = state_info['grad_norm_element/'+param_name] / torch.abs(prev_grad).mean(dtype=torch.float32).item()
                    if param_name in prev_momentum_dict.keys():
                        prev_momentum = prev_momentum_dict[param_name]
                        state_info['momentum_relative_error/'+param_name] = torch.norm(momentum - prev_momentum) / state_info['momentum_norm/'+param_name]
                        state_info['momentum_relative_error_element/'+param_name] = torch.abs(momentum - prev_momentum).mean(dtype=torch.float32).item() / state_info['momentum_norm_element/'+param_name]
                        state_info['momentum_cosine_sim/'+param_name] = cos_func(momentum.view(-1), prev_momentum.view(-1))
                        state_info['momentum_norm_ratio/'+param_name] = state_info['momentum_norm/'+param_name]/ torch.norm(prev_momentum).item()
                        state_info['momentum_norm_element_ratio/'+param_name] = state_info['momentum_norm_element/'+param_name] / torch.abs(prev_momentum).mean(dtype=torch.float32).item()
                    if param_name in prev_grad_dict.keys() and param_name in prev_momentum_dict.keys():
                        update = momentum.clone().mul_(args.momentum).add(grad, alpha=1 - args.momentum).sign_()
                        prev_update = prev_momentum.clone().mul_(args.momentum).add(prev_grad, alpha=1 - args.momentum).sign_()
                        state_info['update_relative_error/'+param_name] = torch.norm(update - prev_update) / torch.norm(update)
                        state_info['update_relative_error_element/'+param_name] = torch.abs(update - prev_update).mean(dtype=torch.float32).item() / torch.abs(update).mean(dtype=torch.float32).item()
                        state_info['update_cosine_sim/'+param_name] = cos_func(update.view(-1), prev_update.view(-1))
                        state_info['update_norm_ratio/'+param_name] = torch.norm(update)/ torch.norm(prev_update).item()
                        state_info['update_norm_element_ratio/'+param_name] = torch.abs(update).mean(dtype=torch.float32).item() / torch.abs(prev_update).mean(dtype=torch.float32).item()
                
            # Save Previous Information
            if num_updates % args.log_interval == args.log_interval-args.state_interval:
                if args.log_optimizer_state:
                    for p in optimizer.state.keys():
                        param_name = param_name_dict[p]
                        prev_weight_dict[param_name] = p.data.detach().clone()
                        prev_grad_dict[param_name] = p.grad.detach().clone()
                        prev_momentum_dict[param_name] = optimizer.state[p]["exp_avg"].detach().clone()

            if str(num_updates) in args.log_weight_iters:
                if args.log_wandb:
                    log_dict = {
                        "iter": num_updates,
                    }
                    for p in optimizer.state.keys():
                        param_name = param_name_dict[p]
                        grad = p.grad
                        momentum = optimizer.state[p]["exp_avg"]
                        log_dict['gradient/'+param_name] = wandb.Histogram(grad.cpu().detach().numpy())
                        log_dict['momentum/'+param_name] = wandb.Histogram(momentum.cpu().detach().numpy())
                    wandb.log(log_dict, step=num_updates)

        if last_batch or num_updates % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.log_wandb:
                    log_dict = {'epoch' : epoch, 'iter': num_updates, 'lr': lr, 'loss':losses_m.val}
                    if state_info is not None:
                        log_dict.update(state_info)
                    wandb.log(log_dict)
                if math.isnan(losses_m.val):
                    break

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

def create_max_eigen_list(eig_dict):
    eig_list = []
    for dic1 in eig_dict.values():
        for dic2 in dic1.values():
            for eig in dic2.values():
                if eig is not None:
                    eig_list.append(eig)
    return eig_list

if __name__ == '__main__':
    main()

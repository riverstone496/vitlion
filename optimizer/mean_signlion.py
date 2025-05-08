from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.distributed import all_reduce, ReduceOp
import time, os

# functions
def exists(val):
    return val is not None

def mean_quantize(tensor, max_level):
    """
    与えられたテンソルを量子化し、絶対値の平均値が量子化後も保持されるようにスケーリングし、
    上にはみ出た値をクリップして量子化します。

    Parameters:
    tensor (torch.Tensor): 量子化する入力テンソル
    levels (int): 量子化レベルの数

    Returns:
    torch.Tensor: 量子化されたテンソル
    """
    mean_val = torch.mean(torch.abs(tensor))
    scaled_tensor = tensor / mean_val
    quantized_tensor = torch.round(scaled_tensor * (max_level / 2))
    quantized_tensor = quantized_tensor.clip(-max_level, max_level)
    return quantized_tensor

# update functions
def compute_updates_sign_int8(params, grads, exp_avgs, lr, wd, beta1, s, clip):
    """
    Compute the sign-based updates for all parameters and aggregate them into a single vector.
    """
    updates_sign = []
    for p, grad, exp_avg in zip(params, grads, exp_avgs):
        # Step weight decay
        p.data.mul_(1 - lr * wd)
        # Compute update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
        if clip > 0:
            update = torch.clamp(update, max=clip)
        update_sign = mean_quantize(update, s).to(torch.int8)
        updates_sign.append(update_sign.flatten())
    # Concatenate all updates into a single vector
    update_vector_sign = torch.cat(updates_sign)
    return update_vector_sign

# class
class  MeanQuantSignLion(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-4, 
        betas: Tuple[float, float] = (0.9, 0.99), 
        weight_decay: float = 0.0, 
        clip: float = -1.0, 
        use_triton: bool = False
    ):
        assert lr > 0.0, "Learning rate must be positive."
        assert all([0.0 <= beta <= 1.0 for beta in betas]), "Betas must be between 0 and 1."
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.beta1, self.beta2 = betas
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn
        else:
            self.update_fn = None  # Not used in optimized step
        self.ngpus = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
        self.s = int(127 / self.ngpus)
        self.clip = clip

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        # Process each parameter group with its own lr and weight_decay
        self.elapsed_time = 0.0
        total_params = 0
        for group in self.param_groups:
            group_lr = group.get('lr', self.defaults['lr'])
            group_wd = group.get('weight_decay', self.defaults['weight_decay'])
            beta1 = self.beta1
            beta2 = self.beta2

            # Collect this group's parameters, grads, and exp_avgs
            params = []
            grads = []
            exp_avgs = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avgs.append(state['exp_avg'])

            if not params:
                continue

            total_params += sum(p.numel() for p in params)
            original_dtype = params[0].grad.dtype

            # Compute updates for this group
            update_vector = compute_updates_sign_int8(params, grads, exp_avgs, group_lr, group_wd, beta1, s=self.s, clip=self.clip)

            # Distributed all-reduce if needed
            if torch.distributed.is_initialized():
                torch.cuda.synchronize()
                start_time = time.time()
                all_reduce(update_vector, op=ReduceOp.SUM)
                torch.cuda.synchronize()
                self.elapsed_time += time.time() - start_time
                update_vector = update_vector.sign_()

            # Apply updates back to this group's parameters
            idx = 0
            for p, exp_avg in zip(params, exp_avgs):
                numel = p.numel()
                update = update_vector[idx:idx + numel].view_as(p).to(original_dtype)
                idx += numel
                p.add_(update, alpha=-group_lr)
                exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)

        self.numel = total_params
        return loss

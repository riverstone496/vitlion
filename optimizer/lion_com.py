from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.distributed import all_reduce, ReduceOp
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# functions
def exists(val):
    return val is not None

# update functions
def compute_updates_sign(params, grads, exp_avgs, lr, wd, beta1, beta2):
    """
    Compute the sign-based updates for all parameters and aggregate them into a single vector.
    """
    updates = []
    for p, grad, exp_avg in zip(params, grads, exp_avgs):
        # Step weight decay
        p.data.mul_(1 - lr * wd)
        # Compute update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
        p.add_(update, alpha=-lr)
        # decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# class
class LionCom(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-4, 
        betas: Tuple[float, float] = (0.9, 0.99), 
        weight_decay: float = 0.0, 
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

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        
        params = []
        grads = []
        exp_avgs = []
        lr = self.defaults['lr']
        wd = self.defaults['weight_decay']
        beta1 = self.beta1
        beta2 = self.beta2

        # Collect parameters, gradients, and exponential averages
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                    exp_avgs.append(state['exp_avg'])

        if not params:
            return loss  # No parameters to update

        # 変更点開始: 勾配を事前に all_reduce する
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # 同期を確実にするために CUDA 操作を同期
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 各勾配テンソルに対して all_reduce を実行
            for grad in grads:
                all_reduce(grad, op=ReduceOp.SUM)
                grad.div_(world_size)  # 平均勾配にする
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
        else:
            elapsed_time = 0.0
        # 変更点終了
        self.elapsed_time = elapsed_time
        # Compute updates for all parameters and aggregate into a single vector
        compute_updates_sign(params, grads, exp_avgs, lr, wd, beta1, beta2)
        return loss
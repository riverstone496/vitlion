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
def compute_updates_sign(params, grads, exp_avgs, lr, wd, beta1):
    """
    Compute the sign-based updates for all parameters and aggregate them into a single vector.
    """
    updates = []
    for p, grad, exp_avg in zip(params, grads, exp_avgs):
        # Step weight decay
        p.data.mul_(1 - lr * wd)
        # Compute update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
        updates.append(update.flatten())
    # Concatenate all updates into a single vector
    update_vector = torch.cat(updates)
    return update_vector

# class
class LionComBF16(Optimizer):
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
                grad = grad.to(torch.bfloat16)
                all_reduce(grad, op=ReduceOp.SUM)
                grad = grad.to(torch.float32)
                grad.div_(world_size)  # 平均勾配にする
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
        else:
            elapsed_time = 0.0
        # 変更点終了

        # Compute updates for all parameters and aggregate into a single vector
        update_vector = compute_updates_sign(params, grads, exp_avgs, lr, wd, beta1)

        # 変更点開始: update_vector に対する all_reduce を削除
        # update_vector の all_reduce を行わないため、この部分を削除またはコメントアウトします。
        # if torch.distributed.is_initialized():
        #     torch.cuda.synchronize()  # Ensure all previous operations are complete
        #     start_time = time.time()
        #     all_reduce(update_vector, op=ReduceOp.SUM)
        #     torch.cuda.synchronize()  # Ensure all_reduce is complete
        #     elapsed_time = time.time() - start_time
        #     # Normalize the update if necessary (e.g., divide by world size)
        #     world_size = torch.distributed.get_world_size()
        #     update_vector /= world_size
        # else:
        #     elapsed_time = 0.0
        # 変更点終了

        # Apply the updates back to each parameter
        idx = 0  # Pointer to traverse the update_vector
        for p, exp_avg in zip(params, exp_avgs):
            numel = p.numel()
            # Extract the corresponding update for this parameter
            update = update_vector[idx:idx + numel].view_as(p)
            idx += numel
            # Apply the update
            p.add_(update, alpha=-lr)
            # Update the exponential moving average
            exp_avg_prev = exp_avg.clone().detach()
            exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)

        self.elapsed_time = elapsed_time
        self.numel = idx
        return loss
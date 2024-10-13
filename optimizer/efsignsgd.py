from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

# functions
def exists(val):
    return val is not None

# update functions
def update_fn(p, grad, exp_avg, memory, lr, wd, beta1):
    # stepweight decay
    p.data.mul_(1 - lr * wd)
    # weight update
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    update = exp_avg.clone()
    pt = (update + memory).sign_()
    p.add_(pt, alpha=-lr)
    memory.copy_(update).add_(pt, alpha=-1)
    # decay the momentum running average coefficient

# class
class EFSignSGD(Optimizer):
    def __init__(
        self, params, lr: float = 1e-4, momentum: float = 0.9, weight_decay: float = 0.0, use_triton: bool = False
    ):
        assert lr > 0.0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_fn = update_fn
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, momentum, state = p.grad, group["lr"], group["weight_decay"], group["momentum"], self.state[p]
                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["memory"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                memory = state["memory"]
                update_fn(p, grad, exp_avg, memory, lr, wd, momentum)
        return loss

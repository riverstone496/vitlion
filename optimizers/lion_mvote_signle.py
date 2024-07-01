from typing import Callable, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer

# functions
def exists(val):
    return val is not None

# update functions
def make_update(p, grad, exp_avg, update, lr, wd, beta1, beta2):
    p.data.mul_(1 - lr * wd)
    update.add_(exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_())
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

def update_fn(p, update, lr):
    update = update.sign_()
    p.add_(update, alpha=-lr)
    # decay the momentum running average coefficient
    update.mul_(0)

class Lion_mvote(Optimizer):
    def __init__(
        self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, num_clients: int = 1, use_triton: bool = False
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.num_clients = num_clients
        self.update_fn = update_fn
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def make_update_vec(self, client_num, closure: Optional[Callable] = None):
        assert 0 <= client_num < self.num_clients, "Invalid client number"
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = p.grad, group["lr"], group["weight_decay"], *group["betas"], self.state[p]
                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = [torch.zeros_like(p) for _ in range(self.num_clients)]
                    state["update"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"][client_num]
                update = state["update"]
                make_update(p, grad, exp_avg, update, lr, wd, beta1, beta2)
    
    @torch.no_grad()
    def sync_update(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    exp_avg_list = state["exp_avg"]
                    avg_exp_avg = torch.mean(torch.stack(exp_avg_list), dim=0)
                    state["exp_avg"] = [avg_exp_avg.clone() for _ in range(self.num_clients)]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = p.grad, group["lr"], group["weight_decay"], *group["betas"], self.state[p]
                update = state["update"]
                update_fn(p, update, lr)
        return loss

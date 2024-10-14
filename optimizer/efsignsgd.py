from typing import Callable, Optional
import torch
from torch.optim.optimizer import Optimizer

# ヘルパー関数
def exists(val):
    return val is not None

# 更新関数
def update_fn(p, grad, exp_avg, memory, lr, wd, beta1):
    # weight decayを勾配に適用
    if wd != 0:
        grad = grad + wd * p.data
    # モメンタムの更新
    exp_avg.mul_(beta1).add_(grad, alpha=1)
    # 更新ステップの計算
    d_p = lr * exp_avg.clone()
    corrected_gradient = lr * (d_p + memory).sign_()
    # パラメータの更新
    p.add_(corrected_gradient, alpha=-1)
    # メモリの更新
    memory.add_(d_p).add_(corrected_gradient, alpha=-1)

# EFSignSGDクラス
class EFSignSGD(Optimizer):
    def __init__(
        self, params, lr: float = 1e-4, momentum: float = 0.9, weight_decay: float = 0.0, use_triton: bool = False
    ):
        assert lr > 0.0, "Learning rate must be positive"
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
                # 状態の初期化
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["memory"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                memory = state["memory"]
                # 更新関数の適用
                update_fn(p, grad, exp_avg, memory, lr, wd, momentum)
        return loss
from typing import Callable, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer

# ヘルパー関数
def exists(val):
    return val is not None

# 更新関数
def update_fn(p, grad, exp_avg, memory, lr, wd, beta1, beta2):
    # Weight Decayを勾配に直接適用
    if wd != 0:
        grad = grad + wd * p.data
    
    # モメンタムの更新
    d_p = lr * exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1)
    
    # 圧縮された勾配の計算
    corrected_gradient = lr * (d_p + memory).sign_()
    
    # パラメータの更新
    p.add_(corrected_gradient, alpha=-1)
    
    # モメンタムの指数移動平均の更新
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
    
    # メモリの更新
    memory.add_(d_p).add_(corrected_gradient, alpha=-1)

# EfLionクラス
class EfLion(Optimizer):
    def __init__(
        self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False
    ):
        assert lr > 0.0, "Learning rate must be positive"
        assert all([0.0 <= beta <= 1.0 for beta in betas]), "Betas must be in [0, 1]"
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad = p.grad
                state = self.state[p]
                # 状態の初期化
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["memory"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                memory = state["memory"]
                # 更新関数の適用
                self.update_fn(p, grad, exp_avg, memory, lr, wd, beta1, beta2)
        return loss
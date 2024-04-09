from collections import defaultdict
import torch

def average_gradients(models, global_model):
    """
    複数のモデルの勾配を平均し、その平均勾配をグローバルモデルの勾配として設定する。
    
    :param models: 勾配を持つ複数のPyTorchモデルのリスト
    :param global_model: 平均勾配を受け取るグローバルモデル
    """
    # パラメータごとに勾配を平均するための辞書を初期化
    avg_grads = defaultdict(torch.Tensor)

    # 各モデルの勾配を集計
    for model in models:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if avg_grads[name].nelement() == 0:
                    # 最初のモデルの勾配をそのまま利用
                    avg_grads[name] = param.grad.clone()
                else:
                    # 既存の勾配に加算
                    avg_grads[name] += param.grad

    # 勾配の平均を計算
    for name in avg_grads:
        avg_grads[name] /= len(models)

    # 平均勾配をグローバルモデルに設定
    for name, param in global_model.named_parameters():
        if name in avg_grads:
            param.grad = avg_grads[name].clone()

def average_momentums(models):
    """
    複数のモデルの勾配を平均し、その平均勾配をグローバルモデルの勾配として設定する。
    
    :param models: 勾配を持つ複数のPyTorchモデルのリスト
    :param global_model: 平均勾配を受け取るグローバルモデル
    """
    # パラメータごとに勾配を平均するための辞書を初期化
    avg_grads = defaultdict(torch.Tensor)
    # 各モデルの勾配を集計
    for model in models:
        for name, param in model.named_parameters():
            if param.exp_avg is not None:
                if avg_grads[name].nelement() == 0:
                    # 最初のモデルの勾配をそのまま利用
                    avg_grads[name] = param.exp_avg.clone()
                else:
                    # 既存の勾配に加算
                    avg_grads[name] += param.exp_avg
    # 勾配の平均を計算
    for name in avg_grads:
        avg_grads[name] /= len(models)
    
    for model in models:
        for name, param in model.named_parameters():
            if param.exp_avg is not None:
                param.exp_avg = avg_grads[name]


def apply_lion_preprocessing(model, betas, sign = True):
    """
    Lion optimizerに基づく勾配の前処理を行い、更新ベクトルを新しい勾配として設定する関数。
    :param model: 勾配を持つPyTorchモデル
    :param lr: 学習率
    :param betas: モメンタム項 (beta1, beta2)
    :param weight_decay: 重み減衰パラメータ
    """
    beta1, beta2 = betas

    # モデルの各パラメータに対して処理を行う
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data
                if 'exp_avg' not in param.__dict__:
                    param.exp_avg = torch.zeros_like(param.data)
                exp_avg = param.exp_avg
                # Lion optimizerの更新関数を使用して勾配を更新
                # 重み減衰
                # 勾配の更新
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                if sign:
                    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                else:
                    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                param.grad.data = update


def sign_gradients(model):
    # モデルの各パラメータに対して処理を行う
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data
                param.grad.data = grad.sign_()
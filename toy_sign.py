import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
from optimizers.lion import Lion

# Custom Sign SGD with momentum optimizer
class SignSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                p.data.add_(-lr * torch.sign(buf))

        return loss

def generate_data(n, d):
    y = torch.randint(0, 2, (n,)).float() * 2 - 1
    A = torch.zeros((n, d))
    for i in range(n):
        A[i, 0] = y[i]
        A[i, 1:4] = 1
        for j in range(4 + 5 * (i + 1), 4 + 5 * (i + 1) + 2 * (1 + int(y[i].item()))):
            if j < d:
                A[i, j] = 1
    return A, y

class Model(nn.Module):
    def __init__(self, d):
        super(Model, self).__init__()
        self.linear = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

def train_model(args):
    n, d = args.num_samples, int(args.dimension_samples * args.num_samples)
    A, y = generate_data(n, d)
    y = y.view(-1, 1)

    # Split into train and test
    indices = torch.randperm(n)
    train_indices, test_indices = indices[:n//2], indices[n//2:]
    A_train, y_train = A[train_indices], y[train_indices]
    A_test, y_test = A[test_indices], y[test_indices]

    model = Model(d)
    criterion = nn.MSELoss()

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == 'lion':
        optimizer = Lion(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2))
    elif args.optim == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'sign_sgd':
        optimizer = SignSGD(model.parameters(), lr=args.lr, momentum = 0)
    elif args.optim == 'sign_sgd_momentum':
        optimizer = SignSGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("Invalid optimizer choice")

    if args.log_wandb:
        wandb.init(project="optimization-experiment", config=args)
        wandb.watch(model, log="all")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(A_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(A_test)
            test_loss = criterion(test_outputs, y_test).item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}")

        if args.log_wandb:
            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimization Experiment")
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer choice')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--beta2', type=float, default=0.1, help='Momentum')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--dimension_samples', type=float, default=6, help='data vs parameters')
    parser.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
    args = parser.parse_args()
    train_model(args)

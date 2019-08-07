import torch
from torch import nn


class BatchNorm(nn.Module):
    '''custom implement batch normalization with autograd by Antinomy
    '''

    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        # auxiliary parameters
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1
        # hyper paramaters
        self.gamma = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)      
        # moving_averge
        self.moving_mean = torch.zeros(self.num_features)
        self.moving_var = torch.ones(self.num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.moving_var)
        nn.init.zeros_(self.moving_mean)

    def forward(self, X):
        assert len(X.shape) in (2, 4)
        if X.device.type != 'cpu':
            self.moving_mean = self.moving_mean.cuda()
            self.moving_var = self.moving_var.cuda()
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,
                                         self.moving_mean, self.moving_var,
                                         self.training, self.eps, self.momentum)
        return Y


def batch_norm(X, gamma, beta, moving_mean, moving_var, is_training=True, eps=1e-5, momentum=0.9,):

    if len(X.shape) == 2:
        mu = torch.mean(X, dim=0)
        var = torch.mean((X - mu) ** 2, dim=0)
        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mu
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        out = gamma * X_hat + beta

    elif len(X.shape) == 4:
        shape_2d = (1, X.shape[1], 1, 1)
        mu = torch.mean(X, dim=(0, 2, 3)).view(shape_2d)
        var = torch.mean(
            (X - mu) ** 2, dim=(0, 2, 3)).view(shape_2d) # biased
        X_hat = (X - mu) / torch.sqrt(var + eps)
        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean.view(shape_2d) + (1.0 - momentum) * mu
            moving_var = momentum * moving_var.view(shape_2d) + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean.view(shape_2d)) / torch.sqrt(moving_var.view(shape_2d) + eps)

        out = gamma.view(shape_2d) * X_hat + beta.view(shape_2d)

    return out, moving_mean, moving_var

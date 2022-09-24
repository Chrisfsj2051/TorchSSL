import torch
import torch.nn as nn
from torch import Tensor


class Dropout(nn.Module):

    def __init__(self, keep_prob=0.5) -> None:
        super(Dropout, self).__init__()
        self.keep_prob = keep_prob

    def forward(self, input, training=True):
        """ .
        Args:
            input: [B, C, H, W]
        """
        if training:
            mask = torch.rand(input.shape) > (1 - self.keep_prob)
            ratio = torch.size() / mask.sum()
            input = input * mask * ratio
        return input


class EMA(nn.Module):
    def __init__(self, warmup_iter = 1000, momentum=0.999):
        self.momentum = momentum
        self.iter_cnt = 0
        self.warmup_iter = warmup_iter

    def compute_lr(self):
        if self.iter_cnt > self.warmup_iter:
            return self.momentum
        return self.momentum * (self.warmup_iter - self.iter_cnt) / self.warmup_iter + (1 - self.momentum)

    def forward(self, model_params, ema_model_params):
        self.iter_cnt += 1
        for key in model_params:
            lr = self.compute_lr()
            ema_model_params[key] = ((1 - lr) * model_params[key] +
                                     lr * ema_model_params[key])
        return ema_model_params
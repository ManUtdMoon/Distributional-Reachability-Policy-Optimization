import abc
import torch
import torch.nn.functional as F
import numpy as np


class SafeEnv(abc.ABC):
    @abc.abstractmethod
    def check_violation(self, states: np.array):
        pass

    @abc.abstractmethod
    def check_done(self, states: np.array):
        pass


def nonneg_barrier(x):
    return F.softplus(-3 * x)


def interval_barrier(x, lb, rb, eps=1e-2, grad=None):
    x = (x - lb) / (rb - lb) * 2 - 1
    b = -((1 + x + eps) * (1 - x + eps) / (1 + eps)**2).log()
    b_min, b_max = 0, -np.log(eps * (2 + eps) / (1 + eps)**2)
    if grad is None:
        grad = 2. / eps / (2 + eps)
    out = grad * (abs(x) - 1)
    return torch.where(torch.as_tensor((-1 < x) & (x < 1)), b / b_max, 1 + out)

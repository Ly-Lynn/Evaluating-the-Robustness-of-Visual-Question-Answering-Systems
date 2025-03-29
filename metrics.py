import torch.nn as nn
import torch

def L2(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=1)).mean()
import numpy as np
import torch


class TransformNormalDistribution(object):
    def __init__(self, var, mean):
        self.var = var
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.var
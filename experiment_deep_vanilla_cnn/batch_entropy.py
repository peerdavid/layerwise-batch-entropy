import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy
import scipy.stats
import random


def batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a mini-batch.
    """
    if(x.shape[0] <= 1):
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")

    x = torch.flatten(x, start_dim=1)
    x_std = torch.std(x, dim=0)
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    return torch.mean(entropies)


class LBELoss(nn.Module):
    """ Computation of the LBE + CE loss.
        See also https://www.wolframalpha.com/input/?i=%28%28x-0.8%29*0.5%29**2+for+x+from+0+to+2+y+from+0+to+0.5
    """
    def __init__(self, num, lbe_alpha=0.5, lbe_alpha_min=0.5, lbe_beta=0.5):
        super(LBELoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()
        lbe_alpha = torch.ones(num) * lbe_alpha
        self.lbe_alpha_p = torch.nn.Parameter(lbe_alpha, requires_grad=True)
        self.lbe_alpha_min = torch.FloatTensor([lbe_alpha_min]).to("cuda")
        self.lbe_beta = lbe_beta

    def lbe_per_layer(self, a, i):
        lbe_alpha_l = torch.abs(self.lbe_alpha_p[i])
        lbe_l = (batch_entropy(a)-torch.maximum(self.lbe_alpha_min, lbe_alpha_l))**2
        return lbe_l * self.lbe_beta

    def __call__(self, output, target):
        output, A = output
        ce = self.ce(output, target)

        if A is None:
            return ce, ce, torch.zeros(1)

        losses = [self.lbe_per_layer(a, i) for i, a in enumerate(A)]
        lbe = torch.mean(torch.stack(losses)) * ce
        return ce+lbe, ce, lbe


class CELoss(nn.Module):
    """ Wrapper around the Cross Entropy loss to be compatible with models
        that output intermediate results.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lbe_alpha_p = torch.zeros(1)
        self.lbe_beta = torch.zeros(1)

    def __call__(self, output, target):
        output, A = output
        ce = self.ce(output, target)
        return ce, ce, 0.0

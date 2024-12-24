import math
import torch
from torch import nn
from torch.autograd import Variable


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class L0Network(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(L0Network, self).__init__()
        self.layer0 = L0GateLayer(input_size)

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, z, mask = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.output(out)
        return out, z, mask


class L0Gate(nn.Module):
    def __init__(self, gate_size, loc_mean=1, loc_sdev=0.01, beta=2 / 3, gamma=-0.1, zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(L0Gate, self).__init__()
        self.gate_size = gate_size
        self.loc = nn.Parameter(torch.zeros((self.gate_size,)).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros((self.gate_size,)))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


class L0GateLayer(L0Gate):
    def __init__(self, gate_size):
        super(L0GateLayer, self).__init__(gate_size)

    def forward(self, input):
        mask, penalty = self._get_mask()
        return input * mask, penalty, mask

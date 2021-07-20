import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

lass RG_Conv(nn.Module):
    def __init__(self, RG, K, task_num, c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.RG = RG
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        w = h = kernel_size
        self.w_shape = (c_out, c_in, w, h)

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(c_out, c_in, w, h))) # あとでinit

        if self.RG:
            scale = 1e-1
            self.LM_list = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.Tensor(w * c_in, K)) * scale) for _ in range(task_num)])
            self.RM_list = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.Tensor(K, h * c_out)) * scale) for _ in range(task_num)])

            self.LM_list[0] = nn.Parameter(torch.zeros(w * c_in, K))
            self.RM_list[0] = nn.Parameter(torch.zeros(K, h * c_out))
        
    def forward(self, x, task: int):
        if self.RG:
            R = torch.mm(self.LM_list[task], self.RM_list[task]).view(self.w_shape)
            weight = R + self.weight
        else:
            weight = self.weight

        return nn.functional.conv2d(
            x, weight=weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
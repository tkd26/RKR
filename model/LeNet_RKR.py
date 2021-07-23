import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class RG_Conv(nn.Module):
    def __init__(self, RG, K, task_num, c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.RG = RG
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.w = self.h = kernel_size
        self.c_in = c_in
        self.c_out = c_out

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.c_out, self.c_in, self.w, self.h)))

        if self.RG:
            self.scale = 1e-1
            self.LM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.w * self.c_in, K)) * self.scale) for _ in range(task_num)])
            self.RM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, self.h * self.c_out)) * self.scale) for _ in range(task_num)])
            # self.M_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.c_out, self.c_in, self.w, self.h)) * self.scale) for _ in range(task_num)])
        
    def forward(self, x, task: int):
        # print(task)
        if self.RG:
            R = torch.matmul(self.LM_list[task], self.RM_list[task]).view(self.w, self.h, self.c_in, self.c_out)
            R = R.permute(3, 2, 0, 1)
            # R = self.M_list[task]
            weight = R + self.weight
        else:
            weight = self.weight

        return nn.functional.conv2d(
            x, weight=weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class RG_FC(nn.Module):
    def __init__(self, RG, K, task_num, h_in, h_out, bias=False):
        super().__init__()

        self.RG = RG
        self.h_in = h_in
        self.h_out = h_out

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.h_out, self.h_in)))

        if self.RG:
            self.scale = 1e-1
            self.LM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.h_in, K)) * self.scale) for _ in range(task_num)])
            self.RM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, self.h_out)) * self.scale) for _ in range(task_num)])
            # self.M_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.h_out, self.h_in))* self.scale) for _ in range(task_num)])

    def forward(self, x, task: int):
        if self.RG:
            R = torch.matmul(self.LM_list[task], self.RM_list[task])
            R = R.permute(1, 0)
            # R = self.M_list[task]
            weight = R + self.weight
        else:
            weight = self.weight
        return nn.functional.linear(x, weight, bias=None)

class SFG_Conv(nn.Module):
    def __init__(self, c_out, task_num: int):
        super().__init__()
        self.F_list = nn.ParameterList([nn.Parameter(torch.ones(c_out)) for _ in range(task_num)])
        # self.F_list = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.Tensor(c_out))) for _ in range(task_num)])
        # self.F_list[0] = nn.Parameter(torch.ones(1))
    
    def forward(self, x, task):
        F = self.F_list[task].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        F = F.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * F
        return x

class SFG_FC(nn.Module):
    def __init__(self, c_out, task_num: int):
        super().__init__()
        self.F_list = nn.ParameterList([nn.Parameter(torch.ones(c_out)) for _ in range(task_num)])
        # self.F_list = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.Tensor(c_out))) for _ in range(task_num)])
    
    def forward(self, x, task: int):
        F = self.F_list[task]
        F = F.unsqueeze(0)
        F = F.repeat(x.shape[0], 1)
        x *= F
        return x

class LeNet(nn.Module):
    def __init__(self, conf_model):
        super(LeNet, self).__init__()

        self.RG = conf_model['RG']
        self.SFG = conf_model['SFG']

        self.conv1 = RG_Conv(self.RG, conf_model['K'], conf_model['task_num'], 3, 6, 5)
        self.conv2 = RG_Conv(self.RG, conf_model['K'], conf_model['task_num'], 6, 16, 5)
        self.fc1 = RG_FC(self.RG, conf_model['K'], conf_model['task_num'], 16*5*5, 120)
        self.fc2 = RG_FC(self.RG, conf_model['K'], conf_model['task_num'], 120, 84)
        self.fc_list = nn.ModuleList([nn.Linear(84, 10) for _ in range(conf_model['task_num'])])

        if self.SFG:
            self.sfg_conv1 = SFG_Conv(6, conf_model['task_num'])
            self.sfg_conv2 = SFG_Conv(16, conf_model['task_num'])
            self.sfg_fc1 = SFG_FC(120, conf_model['task_num'])
            self.sfg_fc2 = SFG_FC(84, conf_model['task_num'])

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(16*5*5, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 10)

    def forward(self, x, task):
        out = F.relu(self.conv1(x, task))
        if self.SFG: out = self.sfg_conv1(out, task)
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out, task))
        if self.SFG: out = self.sfg_conv2(out, task)
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out, task))
        if self.SFG: out = self.sfg_fc1(out, task)

        out = F.relu(self.fc2(out, task))
        if self.SFG: out = self.sfg_fc2(out, task)

        out = self.fc_list[task](out)
        return out
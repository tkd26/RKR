import sys
import torch
from torch import Tensor
import torch.nn as nn
from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class RG_Conv(nn.Module):
    def __init__(self, weight_shape, K: int, task_num: int):
        super().__init__()
        w, h, c_in, c_out = weight_shape

        scale = 1e-5

        self.LM_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(w * c_in, K)) * scale) for _ in range(task_num)])
        self.RM_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, h * c_out)) * scale) for _ in range(task_num)])
        # self.LM_list = nn.ParameterList([nn.Parameter(torch.zeros(w * c_in, K) + 1e-4) for _ in range(task_num)])
        # self.RM_list = nn.ParameterList([nn.Parameter(torch.zeros(K, h * c_out) + 1e-4) for _ in range(task_num)])

        self.LM_list[0] = nn.Parameter(torch.zeros(w * c_in, K))
        self.RM_list[0] = nn.Parameter(torch.zeros(K, h * c_out))

        # self.LM_list = nn.ParameterList([nn.Parameter(torch.zeros(w * c_in, K)) for _ in range(task_num)])
        # self.RM_list = nn.ParameterList([nn.Parameter(torch.zeros(K, h * c_out)) for _ in range(task_num)])
    
    def forward(self, weight, task: int):
        R = torch.mm(self.LM_list[task], self.RM_list[task])
        R = R.view(weight.shape)
        # print(R[0][0][0])
        R = R + weight
        return R

class RG_FC(nn.Module):
    def __init__(self, weight_shape, K: int, task_num: int):
        super().__init__()
        h_in, h_out = weight_shape

        scale = 1e-5

        self.LM_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(h_out, K)) * scale) for _ in range(task_num)])
        self.RM_list = nn.ParameterList([nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, h_in)) * scale) for _ in range(task_num)])

        self.LM_list[0] = nn.Parameter(torch.zeros(h_out, K))
        self.RM_list[0] = nn.Parameter(torch.zeros(K, h_in))

        # self.LM_list = nn.ParameterList([nn.Parameter(torch.zeros(h_out, K)) for _ in range(task_num)])
        # self.RM_list = nn.ParameterList([nn.Parameter(torch.zeros(K, h_in)) for _ in range(task_num)])

    def forward(self, weight, task: int):
        R = torch.mm(self.LM_list[task], self.RM_list[task])
        R = R.view(weight.shape)
        R = R + weight
        return R

class SFG_Conv(nn.Module):
    def __init__(self, c_out, task_num: int):
        super().__init__()
        self.F_list = nn.ParameterList([nn.Parameter(torch.ones(c_out)) for _ in range(task_num)])
    
    def forward(self, x, task):
        F = self.F_list[task]
        F = F.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        F = F.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x *= F
        return x

class SFG_FC(nn.Module):
    def __init__(self, c_out, task_num: int):
        super().__init__()
        self.F_list = nn.ParameterList([nn.Parameter(torch.ones(c_out)) for _ in range(task_num)])
    
    def forward(self, x, task: int):
        F = self.F_list[task]
        F = F.unsqueeze(0)
        F = F.repeat(x.shape[0], 1)
        x *= F
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)
                    #  padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conf_model = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.RG = conf_model['RG']
        self.SFG = conf_model['SFG']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride 

        if self.RG:
            self.rg1 = RG_Conv([3, 3, inplanes, planes], conf_model['K'], conf_model['task_num'])
            self.rg2 = RG_Conv([3, 3, planes, planes], conf_model['K'], conf_model['task_num'])
            if self.downsample is not None:
                self.rg_down_conv = RG_Conv([1, 1, inplanes, planes * self.expansion], conf_model['K'], conf_model['task_num'])

        if self.SFG:
            self.sfg1 = SFG_Conv(planes, conf_model['task_num'])
            self.sfg2 = SFG_Conv(planes, conf_model['task_num'])
            if self.downsample is not None:
                self.sfg_down_conv = SFG_Conv(planes * self.expansion, conf_model['task_num'])

    def forward(self, x: Tensor, task: int) -> Tensor:
        identity = x

        if self.RG:
            self.conv1.weight.data = self.rg1(self.conv1.weight.data, task=task)

        out = self.conv1(x)
        if self.SFG: 
            out = self.sfg1(out, task=task)
        out = self.bn1(out)
        out = self.relu(out)

        if self.RG:
            self.conv2.weight.data = self.rg2(self.conv2.weight.data, task=task)

        out = self.conv2(out)
        if self.SFG:
            out = self.sfg2(out, task=task)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.RG:
                self.downsample[0].weight.data = self.rg_down_conv(self.downsample[0].weight.data, task=task)
            
            identity = self.downsample[0](x)
            if self.SFG:
                identity = self.sfg_down_conv(identity, task=task)
            identity = self.downsample[1](identity)

            # identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        conf_model,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()

        self.layers = layers
        self.RG = conf_model['RG']
        self.SFG = conf_model['SFG']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if self.RG:
            self.rg_conv = RG_Conv([7, 7, 3, self.inplanes], conf_model['K'], conf_model['task_num'])
            # self.rg_fc = RG_FC([512 * block.expansion, conf_model['class_num']], K, conf_model['task_num'])
        
        if self.SFG:
            self.sfg_conv = SFG_Conv(self.inplanes, conf_model['task_num'])
            # self.sfg_fc = SFG_FC(10, conf_model['task_num'])

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], conf_model=conf_model)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], conf_model=conf_model)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], conf_model=conf_model)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], conf_model=conf_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, conf_model['class_num'])
        self.fc_list = nn.ModuleList([nn.Linear(512 * block.expansion, conf_model['class_num']) for _ in range(conf_model['task_num'])])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, conf_model = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.ModuleList([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conf_model))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conf_model=conf_model))

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, task: int) -> Tensor:

        # See note [TorchScript super()]
        pre_conv1 = self.conv1.weight.clone()

        if self.RG:
            self.conv1.weight.data = self.rg_conv(self.conv1.weight.data, task=task)
        # print(torch.all(pre_conv1==self.conv1.weight))

        x = self.conv1(x)
        if self.SFG:
            x = self.sfg_conv(x, task=task)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(self.layers[0]):
            x = self.layer1[i](x, task)
        for i in range(self.layers[1]):
            x = self.layer2[i](x, task)
        for i in range(self.layers[2]):
            x = self.layer3[i](x, task)
        for i in range(self.layers[3]):
            x = self.layer4[i](x, task)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc_list[task](x)

        # if self.RG:
        #     self.fc.weight = self.rg_fc(self.fc.weight, task=task)
        # x = self.fc(x)
        # if self.SFG:
        #     x = self.sfg_fc(x, task=task)

        return x

    def forward(self, x: Tensor, task: int) -> Tensor:
        return self._forward_impl(x, task)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    conf_model,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, conf_model, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        state_dict = '/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet18-f37072fd.pth'
        model.load_state_dict(torch.load(state_dict))
    return model


def resnet18(
    pretrained: bool = False, progress: bool = True, conf_model = None, **kwargs: Any
    ) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, conf_model,
                   **kwargs)

# def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)


# def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                    **kwargs)


# def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


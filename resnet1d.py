import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers) -> None:
        # input x = 3 x 6 x T
        super().__init__()
        self.inplanes = 96
        self.out_channels = 256

        self.conv1 = nn.Conv2d(3, self.inplanes // 6, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes // 6)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.inner_block_module1 = nn.Sequential(nn.Conv1d(64, self.out_channels, kernel_size=1, padding=0), nn.BatchNorm1d(256))
        self.inner_block_module2 = nn.Sequential(nn.Conv1d(128, self.out_channels, kernel_size=1, padding=0), nn.BatchNorm1d(256))
        self.inner_block_module3 = nn.Sequential(nn.Conv1d(256, self.out_channels, kernel_size=1, padding=0), nn.BatchNorm1d(256))
        self.inner_block_module4 = nn.Sequential(nn.Conv1d(512, self.out_channels, kernel_size=1, padding=0), nn.BatchNorm1d(256))

        self.layer_block_module1 = nn.Sequential(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1), nn.BatchNorm1d(256))
        self.layer_block_module2 = nn.Sequential(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1), nn.BatchNorm1d(256))
        self.layer_block_module3 = nn.Sequential(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1), nn.BatchNorm1d(256))
        self.layer_block_module4 = nn.Sequential(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1), nn.BatchNorm1d(256))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  # inplanes/6 x 6 x T // 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # inplanes/6 x 6 x T // 4
        x = x.view(x.shape[0], -1, x.shape[-1])  # inplanes x T // 4
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        last_inner = self.inner_block_module4(x4)
        results = [self.layer_block_module4(last_inner)]

        for inner, layer, _x in zip(
            [self.inner_block_module3, self.inner_block_module2, self.inner_block_module1],
            [self.layer_block_module3, self.layer_block_module2, self.layer_block_module1],
            [x3, x2, x1],
        ):
            inner_lateral = inner(_x)
            feat_shape = inner_lateral.shape[-1]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer(last_inner))
        results.append(F.max_pool1d(results[-1], 1, 2, 0))
        return results

    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride), nn.BatchNorm1d(planes * block.expansion)
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

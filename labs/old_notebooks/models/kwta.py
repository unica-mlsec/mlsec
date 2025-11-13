# no need to read all this code
# it's just the definition of the k-WTA network

import torch.nn as nn
import torch.nn.functional as F


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify2D_vol(SparsifyBase):

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        size = x.shape[1] * x.shape[2] * x.shape[3]
        k = int(self.sr * size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sparse1 = Sparsify2D_vol(sparsity)
        self.sparse2 = Sparsify2D_vol(sparsity)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.sparse2(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=10):
        super(SparseResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu = nn.ReLU()
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SparseResNet18(sparsities=[0.5, 0.4, 0.3, 0.2]):
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], sparsities)

import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
import os
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt


class ResNet(nn.Module):
    # Networ for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x


activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

        return hook

def main():
    model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3)

    state = torch.load('Features/embeddings/model_checkpoint_epoch_1.pkl', map_location='cpu')
    cur_dict = model.state_dict()
    state.avgpool.register_forward_hook(get_activation('avgpool'))


    for key in state['model_state_dict'].keys():
        print(key)
        # if key in cur_dict:
        #     if (state['model_state_dict'][key].shape == cur_dict[key].shape):
        #         cur_dict[key] = nn.Parameter(state['model_state_dict'][key].data)
        #     else:
        #         print('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
        # else:
        #     print('\n Error: Loaded weight {} not present in current model'.format(key))

if __name__ == '__main__':
    main()
from BiwiDataLoaderResnet import BiwiDatasetResnetLoader
import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
import os
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt

np.random.seed(42)


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


def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
    return model

def load_npz(file):
    data = np.load(file, allow_pickle=True)
    return data

def main():
    batch_size = 32

    dataset = load_npz('../biwi_dataset_main.npz')
    data = BiwiDatasetResnetLoader(dataset)

    if not os.path.exists('Features/embeddings'):
        os.makedirs('Features/embeddings')

    # ResNet50
    model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3)
    load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load Pretrained model
    # checkpoint = torch.load('output/snapshots/model_checkpoint_epoch_10.pkl')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # running_loss = checkpoint['running_loss']

    train_loader = torch.utils.data.DataLoader(dataset=data.ds_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0
                                               )
    epoch_loss = []
    iter_loss = []
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, pose, folder) in enumerate(train_loader):
            images = np.array(images)
            images = np.moveaxis(images, 3, 1)
            images = torch.from_numpy(images)
            images = Variable(images).float()

            label_angles = Variable(pose).float()
            angles = model(images)

            loss = criterion(angles, label_angles)

            running_loss += loss.item()
            iter_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(f'Epoch:{epoch + 1}, Loss:{loss.item()}')

        epoch_loss.append(running_loss / len(data.ds_train))

        print("Saving Model Checkpoint...")

        path = 'Features/embeddings/model_checkpoint_epoch_' + str(epoch + 1) + '.pkl'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': iter_loss,
            'running_loss': epoch_loss
        }, path)

    plt.plot(epoch_loss)
    plt.show()

if __name__ == '__main__':
    main()
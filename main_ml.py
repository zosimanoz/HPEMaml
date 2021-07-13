import PIL.Image
import cv2

from BiwiTaskLoader import BiwiTaskLoader
from meta import MetaLearner
import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
import os
import utils
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt
import bboxutils
from PIL import Image
import cv2 as cv

np.random.seed(42)

class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
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

    def forward(self, x, y):

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

def show_image_grid(img_arr):
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(20, 2.5))
    # for i in range(5):
    for ax, img in zip(axes.ravel(), img_arr):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax.imshow(img)
    plt.show()

def main():
    meta_batch_size = 5
    n_way = 5
    k_shot = 1
    k_query = 1
    meta_lr = 0.001
    num_updates = 5

    dataset = load_npz('biwi_dataset_main.npz')
    data = BiwiTaskLoader(dataset, batch_size=meta_batch_size, n_way=n_way, k_shot=k_shot, k_query=k_query)

    if not os.path.exists('checkpoints/snapshots2'):
        os.makedirs('checkpoints/snapshots2')

    # ResNet50
    model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3)
    load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    meta = MetaLearner(model,(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3), n_way=n_way, k_shot=k_shot, meta_batch_size=meta_batch_size, alpha=0.001, beta=meta_lr, num_updates=num_updates)

    pred_loss = []
    train_loss = []
    train_loss_mae = []
    pred_loss_mae = []

    for episode_num in range(10):
        support_x, support_y, query_x, query_y = data.next('train')
        # im = support_x[episode_num][0]
        # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()

        if episode_num == 0:
            show_image_grid(support_x[0])
            show_image_grid(query_x[0])

        # support_x : [8, 5, 3, 64, 64]
        support_x = torch.from_numpy(support_x).float()
        query_x = torch.from_numpy(query_x).float()
        support_y = torch.from_numpy(support_y).float()
        query_y = torch.from_numpy(query_y).float()

        losses, mae = meta(support_x, support_y, query_x, query_y)
        # mean_train_loss = np.array(losses.item()).mean()
        mean_train_loss = torch.mean(torch.stack(losses))
        mae_mean_train_loss = torch.mean(torch.stack(mae))

        train_loss.append(mean_train_loss)
        train_loss_mae.append(mae_mean_train_loss)

        # mean_test_losses = []
        # mean_mae_losses = []
        # support_x, support_y, query_x, query_y = data.next('test')  # support, query for test
        # support_x = torch.from_numpy(support_x).float()
        # query_x = torch.from_numpy(query_x).float()
        # support_y = torch.from_numpy(support_y).float()
        # query_y = torch.from_numpy(query_y).float()
        #
        # mean_loss, mae = meta.pred(support_x, support_y, query_x, query_y)
        # mean_test_losses.append(mean_loss)
        # mean_mae_losses.append(mae)
        #
        # mean_test_loss = torch.mean(torch.stack(mean_test_losses))
        # mean_test_loss_mae = torch.mean(torch.stack(mean_mae_losses))
        #
        # pred_loss.append(mean_test_loss)
        # pred_loss_mae.append(mean_test_loss_mae)
        #
        # if episode_num % 2 == 0:
        #
        #     print(f'Episode: {episode_num} , Fine tune loss (Outer Loop Loss) :{mean_train_loss}, '
        #           f'Test loss (Inner Loop Loss): {mean_test_loss}, Train MAE: {mae_mean_train_loss}, '
        #           f'Test MAE: {mean_test_loss_mae}')
        #
        #     print("Saving Model Checkpoint...")
        #     path = 'output/snapshots_maml4/model_checkpoint_epoch_' + str(episode_num) + '.pkl'
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'pred_loss': pred_loss,
        #         'train_loss': train_loss,
        #         'pred_loss_mae': pred_loss_mae,
        #         'train_loss_mae': train_loss_mae
        #     }, path)

        if episode_num % 2 == 0:
            mean_test_losses = []
            mean_mae_losses = []

            support_x, support_y, query_x, query_y = data.next('test')  # support, query for test
            support_x = torch.from_numpy(support_x).float()
            query_x = torch.from_numpy(query_x).float()
            support_y = torch.from_numpy(support_y).float()
            query_y = torch.from_numpy(query_y).float()

            print('Predicting')
            show_image_grid(support_x[0].numpy())
            show_image_grid(query_x[0].numpy())

            mean_loss, mae = meta.pred(support_x, support_y, query_x, query_y)
            mean_test_losses.append(mean_loss)
            mean_mae_losses.append(mae)

            mean_test_loss = torch.mean(torch.stack(mean_test_losses))
            mean_test_loss_mae = torch.mean(torch.stack(mean_mae_losses))

            pred_loss.append(mean_test_loss)
            pred_loss_mae.append(mean_test_loss_mae)

            print(f'Episode: {episode_num} , Fine tune loss (Outer Loop Loss) :{mean_train_loss}, '
                  f'Test loss (Inner Loop Loss): {mean_test_loss}, Train MAE: {mae_mean_train_loss}, '
                  f'Test MAE: {mean_test_loss_mae}')

            print("Saving Model Checkpoint...")
            path = 'checkpoints/snapshots2/model_checkpoint_epoch_' + str(episode_num) + '.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'pred_loss': pred_loss,
                'train_loss': train_loss,
                'pred_loss_mae': pred_loss_mae,
                'train_loss_mae': train_loss_mae
            }, path)
            print('Checkpoint Saved')

    plt.plot(pred_loss)
    plt.xlabel("Episodes")
    plt.ylabel("pred_loss")
    plt.show()

    plt.plot(train_loss)
    plt.xlabel("Episodes")
    plt.ylabel("train_loss")
    plt.show()

if __name__ == '__main__':
    main()
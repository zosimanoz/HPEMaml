from BiwiTaskLoaderV2 import BiwiTaskLoaderV2
from BiwiDataLoader import BiwiDatasetLoader
from metaV2 import MetaLearner
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
import torch.nn.functional as F

np.random.seed(42)

#
# class Net(nn.Module):
#     def __init__(self, in_channels=1, num_classes=3):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
#         self.pool = nn.MaxPool2d(2)
#         self.normalize1 = nn.BatchNorm2d(1024)
#         self.normalize2 = nn.BatchNorm2d(512)
#         self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
#         # self.fc1 = nn.Linear(512, 256)
#         self.fc_angles = nn.Linear(512, num_classes)
#
#     def forward(self, x, y):
#         x = F.relu(self.normalize1(self.conv1(x)))
#         # x = self.pool(x)
#         x = F.relu(self.normalize2(self.conv2(x)))
#         # x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc_angles(x)
#
#         return x


class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(Net, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3),
                                 nn.AvgPool2d(kernel_size=2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(64, 64, kernel_size=1),
                                 # nn.AvgPool2d(kernel_size=2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(64, 64, kernel_size=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(64, 64, kernel_size=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, num_classes))


    def forward(self, x, y):
        # x:[5, 1, 28, 28] : 5 way 1 shot
        x = self.net(x)
        # x = x.view(-1, 64)
        x = x.reshape(x.shape[0], -1)
        pred = self.fc(x)

        print(pred)

        return pred


def load_npz(file):
    data = np.load(file, allow_pickle=True)
    return data

def main():
    meta_batch_size = 32
    n_way = 2
    k_shot = 5
    k_query = 5
    meta_lr = 0.01
    num_updates = 5

    data = BiwiTaskLoaderV2('test', batch_size=meta_batch_size, n_way=n_way, k_shot=k_shot, k_query=k_query)

    if not os.path.exists('output/snapshots_maml_v2'):
        os.makedirs('output/snapshots_maml_v2')


    meta = MetaLearner(Net,(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3), n_way=n_way, k_shot=k_shot, meta_batch_size=meta_batch_size, alpha=0.01, beta=meta_lr, num_updates=num_updates)

    pred_loss = []
    train_loss = []
    train_loss_mae = []
    pred_loss_mae = []
    # mean_train_loss = []
    # mean_test_loss = []
    # losses = []
    # mean_loss = []

    for episode_num in range(25):
        support_x, support_y, query_x, query_y = data.next('train')

        # support_x : [8, 5, 3, 64, 64]
        support_x = torch.from_numpy(support_x).float()
        query_x = torch.from_numpy(query_x).float()
        support_y = torch.from_numpy(support_y).float()
        query_y = torch.from_numpy(query_y).float()

        # support_x = support_x.unsqueeze(3).unsqueeze(4)
        # query_x = query_x.unsqueeze(3).unsqueeze(4)



        losses, mae = meta(support_x, support_y, query_x, query_y)
        # mean_train_loss = np.array(losses.item()).mean()
        mean_train_loss = torch.mean(torch.stack(losses))
        mae_mean_train_loss = torch.mean(torch.stack(mae))

        train_loss.append(mean_train_loss)
        train_loss_mae.append(mae_mean_train_loss)


        mean_test_losses = []
        mean_mae_losses = []
        support_x, support_y, query_x, query_y = data.next('test')  # support, query for test
        support_x = torch.from_numpy(support_x).float()
        query_x = torch.from_numpy(query_x).float()
        support_y = torch.from_numpy(support_y).float()
        query_y = torch.from_numpy(query_y).float()


        # support_x = support_x.unsqueeze(3).unsqueeze(4)
        # query_x = query_x.unsqueeze(3).unsqueeze(4)

        mean_loss, mae = meta.pred(support_x, support_y, query_x, query_y)
        mean_test_losses.append(mean_loss)
        mean_mae_losses.append(mae)

        mean_test_loss = torch.mean(torch.stack(mean_test_losses))
        mean_test_loss_mae = torch.mean(torch.stack(mean_mae_losses))

        pred_loss.append(mean_test_loss)
        pred_loss_mae.append(mean_test_loss_mae)

        if episode_num % 2 == 0:

            print(f'Episode: {episode_num} , Fine tune loss (Outer Loop Loss) :{mean_train_loss}, '
                  f'Test loss (Inner Loop Loss): {mean_test_loss}, Train MAE: {mae_mean_train_loss}, '
                  f'Test MAE: {mean_test_loss_mae}')

            print("Saving Model Checkpoint...")
            path = 'output/snapshots_maml_v2/model_checkpoint_epoch_' + str(episode_num) + '.pkl'
            torch.save({
                'model_state_dict': meta.state_dict(),
                'pred_loss': pred_loss,
                'train_loss': train_loss,
                'pred_loss_mae': pred_loss_mae,
                'train_loss_mae': train_loss_mae
            }, path)


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
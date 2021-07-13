# from biwi_dataloader import BiwiNShotDataLoader
from biwi_dataloader_face_crop import BiwiNShotDataLoader
from meta import MetaLearner

from torchmeta.utils.data import BatchMetaDataLoader

import torch
import torch.nn as nn
import math
import torchvision
import os


class CNNImagesNet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        super(CNNImagesNet, self).__init__()

        self.inplanes = 64
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

        self.fc_yaw = nn.Linear(512 * 4, num_bins)
        self.fc_pitch = nn.Linear(512 * 4, num_bins)
        self.fc_roll = nn.Linear(512 * 4, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

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

        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll



def main():
    meta_batch_size = 2
    n_way = 5
    k_shot = 1
    k_query = 1
    meta_lr = 1e-3
    num_updates = 5
    num_episodes = 1

    img_size = 28
    num_bins = 66
    total = 0
    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    omni_data = BiwiNShotDataLoader('dataset', batch_size=meta_batch_size, n_way=n_way,
                              k_shot=k_shot, k_query=k_query, img_size=img_size)
    model = CNNImagesNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    meta = MetaLearner(CNNImagesNet, (torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins) , n_way=n_way, k_shot=k_shot, meta_batch_size=meta_batch_size,
                       alpha=0.1, beta=meta_lr, num_updates=num_updates)

    for episode_num in range(num_episodes):
        support_x, support_y, support_cont, query_x, query_y, query_cont = omni_data.get_batch('train')  # support, query for train

        # support_x : [32, 5, 1, 28, 28]
        support_x = torch.from_numpy(support_x).float()
        query_x = torch.from_numpy(query_x).float()
        support_y = torch.from_numpy(support_y).long()
        query_y = torch.from_numpy(query_y).long()
        support_cont = torch.from_numpy(support_cont).long()
        query_cont = torch.from_numpy(query_cont).long()
        meta(support_x, support_y, support_cont, query_x, query_y, query_cont)
        # loss_y, loss_p, loss_r = meta(support_x, support_y, support_cont, query_x, query_y, query_cont)

        if episode_num % 2 == 0:

            support_x, support_y, support_cont, query_x, query_y, query_cont = omni_data.get_batch('test')  # support, query for test
            support_x = torch.from_numpy(support_x).float()
            query_x = torch.from_numpy(query_x).float()
            support_y = torch.from_numpy(support_y).long()
            query_y = torch.from_numpy(query_y).long()
            support_cont = torch.from_numpy(support_cont).long()
            query_cont = torch.from_numpy(query_cont).long()

            meta.pred(support_x, support_y, support_cont, query_x, query_y, query_cont)

            # loss_y_test, loss_p_test, loss_r_test, grad_out = meta.pred(support_x, support_y, support_cont, query_x, query_y, query_cont)

            # print('episode:', episode_num, '\n Fintune losses:%.6f' % loss_y, loss_p, loss_r, '\n Test losses:%.6f' % loss_y_test, loss_p_test, loss_r_test)

            # Save models at numbered epochs.
            print('Taking snapshot...')
            torch.save(meta.state_dict(), 'output/snapshots/model_checkpoint_epoch_' + str(episode_num + 1) + '.pkl')
            torch.save(model.state_dict(), 'output/snapshots/meta_checkpoint_epoch_' + str(episode_num + 1) + '.pkl')

if __name__ == '__main__':
    main()
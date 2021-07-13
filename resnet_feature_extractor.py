import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
from BiwiDataLoader import BiwiDatasetLoader

data_dir = './test'  # train
features_dir = './Resnet_features_test'  # Resnet_features_train

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        # output = self.net.layer3(output)
        # output = self.net.layer4(output)
        # output = self.net.avgpool(output)
        return output


model = net()

def extractor(img, net, pose, folder):

    x = Variable(img, requires_grad=False)
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    full_embedding = [y, pose, folder]

    return full_embedding


def load_npz(file):
    data = np.load(file, allow_pickle=True)
    return data


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()

    meta_batch_size = 64
    n_way = 5
    k_shot = 1
    k_query = 1

    dataset = load_npz('BIWI_noTrack2.npz')
    data = BiwiDatasetLoader(dataset, batch_size=meta_batch_size, n_way=n_way, k_shot=k_shot, k_query=k_query)

    if not os.path.exists('output/resnet'):
        os.makedirs('output/resnet')

    train_loader = torch.utils.data.DataLoader(dataset=data.train_set,
                                               batch_size=meta_batch_size,
                                               shuffle=True,
                                               num_workers=0
                                               )
    features = []

    # Iterate each image
    for i, (image, pose, folder) in enumerate(train_loader):
        image = np.array(image)
        image = np.moveaxis(image, 3, 1)
        image = torch.from_numpy(image)
        image = Variable(image).float()

        embeddings = extractor(image, model, pose, folder)

        print(embeddings[0].shape, embeddings[0][0].shape)

        features.append(embeddings)

    np.savez('features_10k_resnet.npz', features)

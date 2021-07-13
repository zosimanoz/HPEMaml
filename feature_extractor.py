import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
from BiwiDataLoader import BiwiDatasetLoader

data_dir = './test'  # train
features_dir = './DenseNet_features_test'  # DenseNet_features_train


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        densnet = models.densenet121(pretrained=True)
        self.feature = densnet.features
        self.classifier = nn.Sequential(*list(densnet.classifier.children())[:-1])
        pretrained_dict = densnet.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, x):
        output = self.feature(x)
        avg = nn.AvgPool2d(2, stride=1)
        output = avg(output)
        return output


model = Encoder()


def extractor(img, net, pose, folder):

    x = Variable(img, requires_grad=False)
    print(x.shape)

    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    print(y.shape)

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

    if not os.path.exists('output/densenet'):
        os.makedirs('output/densenet')

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

        features.append(embeddings)

    np.savez('features_10k_densenet.npz', features)

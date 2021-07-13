
import numpy as np


class BiwiDatasetResnetLoader():
    def __init__(self, root):
        self.ds_root = root
        self.ds_train = []

        for img, pose, folder in zip(self.ds_root['image'], self.ds_root['pose'], self.ds_root['folder_name']):
            img_normalized = (img - np.mean(img)) / np.std(img)
            self.ds_train.append([img_normalized, pose, folder])

        # self.train_set = self.ds_train[:10000]
        # self.test_set = self.ds_train[3219:]

    def __getitem__(self, index):
        img, pose, folder = self.ds_train[index]
        return img, pose, folder, self.ds_train[index]

    def __len__(self):
        return len(self.ds_train)

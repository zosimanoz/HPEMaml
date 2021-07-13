import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
import bboxutils
import cv2

def get_filename_list():
    frame_file_names = []
    for (dirpath, dirnames, filenames) in os.walk("../dataset_main"):
        for files in filenames:
            if files.endswith('.png'):
                frame_file_names.append(dirpath + "/" + files)
    return frame_file_names

def get_pose_list(frame_png_images):
    frame_pose_names = []
    for i in range(len(frame_png_images)):
        frame_pose_names.append(frame_png_images[i][0: -7] + "pose.txt")
    return frame_pose_names

def get_depth_images_list(frame_png_images):
    depth_images = []
    for i in range(len(frame_png_images)):
        depth_images.append(frame_png_images[i][0: -7] + "depth.bin")
    return depth_images


class BiwiNShotDataLoader():
    def __init__(self, root, batch_size, n_way, k_shot, k_query, img_size):
        self.resize = img_size
        self.img_ext = ".png"
        self.annot_ext = ".txt"
        self.image_mode = "RGB"

        frame_rgb_path = get_filename_list()
        pose_path = get_pose_list(frame_rgb_path)


        self.rgb_images_paths = frame_rgb_path
        self.poses_path = pose_path
        self.length = len(pose_path)
        self.seed = 42


        if not os.path.isfile(os.path.join(root, 'biwi.npy')):
            temp = dict()

            if not os.path.exists(root):
                os.makedirs(os.path.join(root))

            for (image_path, label_path) in zip(self.rgb_images_paths, self.poses_path):
                euler_angles, cont_labels = self.get_pose_from_path(label_path)
                labels = [euler_angles, cont_labels]
                if image_path in temp:
                    # temp[image_path].append(euler_angles)
                    temp[image_path].append(euler_angles)
                    temp[image_path].append(cont_labels)
                else:
                    # temp[image_path] = euler_angles
                    temp[image_path] = labels

            items = temp.items()
            temp = []  # Free memory
            data = list(items)
            self.x = np.array(data)
            np.save(os.path.join(root, 'biwi.npy'), self.x)
        else:
            self.x = np.load(os.path.join(root, 'biwi.npy'), allow_pickle=True)

        # x : [1623, 20, 1, 28, 28]
        np.random.shuffle(self.x)  # shuffle on the first dim = 1623 cls
        self.x_train, self.x_test = self.x[:14000], self.x[14000:]

        # normalization
        # self.x_train = (self.x_train - np.mean(self.x_train)) / np.std(self.x_train)
        # self.x_test = (self.x_test - np.mean(self.x_test)) / np.std(self.x_test)

        self.batch_size = batch_size
        self.n_class = self.x.shape[0]  # 1623
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    @property
    def images(self):
        return self.X_train

    @property
    def poses(self):
        return self.y_train

    def get_pose_from_path(self, pose_path):
        # Load pose in degrees
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)

        R = np.array(R)
        T = R[3, :]
        R = R[:3, :]
        pose_annot.close()

        R = np.transpose(R)

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        # get binned labels
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        return labels, cont_labels
    def reshape_image(img):
        row,colum =200,200
        try:
            img = cv2.resize(img,row,colum, interpolation=cv2.INTER_AREA)

        except Exception as e:
            print(str(e))
        return img

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        N shot Learning을 한 data batches
        data_pack : [class_num, 20, 1, 28, 28] #class_num : train일 때 1200, test는 423
        return : A list [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """

        dataset_size = self.k_shot * self.n_way
        query_size = self.k_query * self.n_way
        data_cache = []
        offset = 2

        for sample in range(1):  # num of eisodes
            #shape = (200,200,3)
            support_x = np.zeros((self.batch_size, dataset_size, 3, 480, 640))  # [32, 5, 28, 28, 1]
            support_y = np.zeros((self.batch_size, dataset_size, 3))
            support_cont = np.zeros((self.batch_size, dataset_size, 3))

            query_x = np.zeros((self.batch_size, query_size, 3, 480, 640))  # [32, 5, 28, 28, 1]
            query_y = np.zeros((self.batch_size, query_size, 3))
            query_cont = np.zeros((self.batch_size, query_size, 3))

            for i in range(self.batch_size):
                shuffle_idx = np.arange(self.n_way)  # [0,1,2,3,4]
                np.random.shuffle(shuffle_idx)  # [2,4,1,0,3]
                shuffle_idx_test = np.arange(self.n_way)  # [0,1,2,3,4]
                np.random.shuffle(shuffle_idx_test)

                selected_sup_indexes = np.random.choice(data_pack.shape[0], self.n_way, replace=False)
                selected_query_indexes = np.random.choice(data_pack.shape[0], self.n_way, replace=False)

                for j, indexes in enumerate(selected_sup_indexes):
                    image_path, eu_angle = zip(data_pack[indexes])
                    img = Image.open(image_path[0])
                    img = img.convert(self.image_mode)
                    img = np.array(img)
                    img = np.moveaxis(img, 2, 0)
                    img = (img - np.mean(img)) / np.std(img)

                    support_x[i, shuffle_idx[j] * self.k_shot, ...] = img
                    support_y[i, shuffle_idx[j] * self.k_shot] = eu_angle[0][0]
                    support_cont[i, shuffle_idx[j] * self.k_shot] = eu_angle[0][1]

                for j, indexes in enumerate(selected_query_indexes):
                    image_path, eu_angle = zip(data_pack[indexes])
                    img = Image.open(image_path[0])
                    img = img.convert(self.image_mode)
                    img = np.array(img)
                    img = np.moveaxis(img, 2, 0)
                    img = (img - np.mean(img)) / np.std(img)

                    query_x[i, shuffle_idx_test[j] * self.k_query, ...] = img
                    query_y[i, shuffle_idx_test[j] * self.k_query] = eu_angle[0][0]
                    query_cont[i, shuffle_idx_test[j] * self.k_query] = eu_angle[0][1]

            data_cache.append([support_x, support_y, support_cont, query_x, query_y, query_cont])
        return data_cache

    def get_batch(self, mode):
        # mode : train / test
        # Gets next batch from the dataset with name.

        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        # len(self.datasets_cache['train'])) : 100
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
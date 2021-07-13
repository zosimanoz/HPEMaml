import matplotlib.pyplot as plt
import numpy as np
import os


class BiwiTaskLoader():
    def __init__(self, root, batch_size, n_way, k_shot, k_query):

        self.embedding_directory = 'output/encoder'

        temp = dict()
        for (folder, img, pose) in zip(root['folder_name'], root['image'], root['pose']):
            if folder in temp.keys():
                img_normalized = (img - np.mean(img)) / np.std(img)
                temp[folder].append([img_normalized, pose])
            else:
                data = []
                img_normalized = (img - np.mean(img)) / np.std(img)
                data.append([img_normalized, pose])
                temp[folder] = data

        self.x = []
        for item in temp.items():
            self.x.append(item)

        data_train = []
        data_test = []

        for item in self.x:
            if item[0] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                data_train.append(item[1])
                self.x_train = data_train
            else:
                data_test.append(item[1])
                self.x_test = data_test

        self.batchsz = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_class = 24

        assert (k_shot + k_query) <= 20

        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    def load_data_cache(self, data_pack):
        '''
        Collects several batches data for N-shot Learning
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        '''
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        for sample in range(5):  # num of eisodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []

            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                if len(data_pack) > 9:
                    selected_cls = np.random.choice(15, self.n_way, False)
                else:
                    selected_cls = np.random.choice(9, self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(200, self.k_shot + self.k_query, False)

                    # tm = selected_img[:self.k_shot]
                    # tw = selected_img[self.k_shot:]
                    # cs = data_pack[cur_class]

                    # im = data_pack[cur_class][selected_img[:self.k_shot][0]][0]
                    #
                    # plt.imshow(im)
                    # plt.show()
                    #
                    # im2 = data_pack[cur_class][selected_img[self.k_shot:][0]][0]
                    # plt.imshow(im2)
                    # plt.show()

                    # t = data_pack[cur_class][selected_img[:self.k_shot][0]][0]
                    # q = data_pack[cur_class][selected_img[:self.k_shot][0]][1]

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot][0]][0])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:][0]][0])
                    y_spt.append(data_pack[cur_class][selected_img[:self.k_shot][0]][1])
                    y_qry.append(data_pack[cur_class][selected_img[self.k_shot:][0]][1])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)

                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 224, 224, 3)[perm]
                # im2 = x_spt[0]
                # plt.imshow(im2)
                # plt.show()

                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, 3)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 224, 224, 3)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, 3)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

                # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 224, 224, 3)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz, 3)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 224, 224, 3)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz, 3)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
        return data_cache

    def next(self, mode='train'):

        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch



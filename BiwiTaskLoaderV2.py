
import numpy as np
import os


class BiwiTaskLoaderV2():
    def __init__(self, root, batch_size, n_way, k_shot, k_query):

        embeddings = np.load('features_10k_resnet.npz', allow_pickle=True)

        arr = embeddings['arr_0']

        self.x = []
        self.x_train = []
        self.x_test = []

        for i in range(arr.shape[0]):
            for img, pose, folder in zip(arr[i, 0], arr[i, 1], arr[i, 2]):
                img_normalized = (img - np.mean(img)) / np.std(img)
                self.x.append([img_normalized, pose])

        self.x_train = self.x[2000:]
        self.x_test = self.x[:2000]

        self.batchsz = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_class = 24

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

        for sample in range(100):  # num of eisodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []

            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                selected_indexes = np.random.choice(len(data_pack), self.n_way, replace=False)

                for j, cur_index in enumerate(selected_indexes):
                    selected_img = np.random.choice(len(data_pack), self.k_shot + self.k_query, False)

                    set_images = []

                    if self.k_shot > 1:
                        support_set_idx = selected_img[:self.k_shot]
                        query_set_idx = selected_img[self.k_shot:]

                        for k, idx in enumerate(support_set_idx):
                            x_spt.append(data_pack[idx][0])
                            y_spt.append(data_pack[idx][1].detach().numpy())

                        for k, idx in enumerate(query_set_idx):
                            x_qry.append(data_pack[idx][0])
                            y_qry.append(data_pack[idx][1].detach().numpy())

                    else:
                        x_spt.append(data_pack[selected_img[:self.k_shot][0]][0])
                        x_qry.append(data_pack[selected_img[self.k_shot:][0]][0])
                        y_spt.append(data_pack[selected_img[:self.k_shot][0]][1].detach().numpy())
                        y_qry.append(data_pack[selected_img[self.k_shot:][0]][1].detach().numpy())

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 512, 8, 8)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, 3)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 512, 8, 8)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, 3)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

                # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 512, 8, 8)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz, 3)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 512, 8, 8)
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



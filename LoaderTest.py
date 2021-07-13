import pickle as pl

import matplotlib.pyplot as plt
import numpy as np
import os


def load_data_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["image"], d["pose"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

if __name__ == '__main__':

    b = np.load('biwi_dataset_main.npz', allow_pickle=True, fix_imports=True, encoding='latin1')

    for image, pose in zip(b['image'], b['pose']):
        plt.imshow(image)
        plt.show()


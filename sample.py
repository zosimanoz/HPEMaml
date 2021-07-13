import os
import sys
import numpy as np

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass

def main():
    image_size = 64

    a = np.zeros((5, 2048))
    print(a.shape)

    a.reshape(())

if __name__ == '__main__':
    main()
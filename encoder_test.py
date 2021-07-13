import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from BiwiTaskLoader import BiwiTaskLoader
import os
import pandas as pd

def get_data():
    temp = dict()

    file_content = np.load('Features.npz', allow_pickle=True)

    b = np.load('BIWI_noTrack2.npz', allow_pickle=True, fix_imports=True, encoding='latin1')

    for image, pose in zip(b['image'], b['pose']):
        X_embedded = TSNE(n_components=2).fit_transform(image.reshape(-1, 1))
        # print(X_embedded.shape)
        # plt.scatter(x=X_embedded[:,0],y=X_embedded[:,1])
        # plt.show()

        df = pd.DataFrame()

        df['tsne-2d-one'] = X_embedded[:, 0]
        df['tsne-2d-two'] = X_embedded[:, 1]

        plt.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1])

        plt.show()


    arr = file_content['arr_0']

    for images, pose, folder in zip(arr[0,0], arr[0,1], arr[0, 2]):
        X_embedded = TSNE(n_components=2).fit_transform(images.reshape(-1,1))
        # print(X_embedded.shape)
        # plt.scatter(x=X_embedded[:,0],y=X_embedded[:,1])
        # plt.show()

        df = pd.DataFrame()

        df['tsne-2d-one'] = X_embedded[:, 0]
        df['tsne-2d-two'] = X_embedded[:, 1]
        #
        # sns.scatterplot(
        #     palette=sns.color_palette("hls", 2),
        #     data=df,
        #     legend="full",
        #     alpha=0.9
        # )
        #
        plt.scatter(x = X_embedded[:, 0], y=X_embedded[:, 1])

        plt.show()




if __name__ == '__main__':
    get_data()
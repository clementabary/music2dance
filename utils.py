import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class StickDataset(Dataset):
    def __init__(self, name, normalize=None):
        self.sticks = stickwise(load_dataset(name))
        self.scaler = None
        if normalize == 'minmax':
            self.scaler = MinMaxScaler()
            dshape = np.shape(self.sticks)
            self.sticks = np.reshape(self.sticks, (dshape[0], -1))
            self.sticks = self.scaler.fit_transform(self.sticks)
            self.sticks = np.reshape(self.sticks, dshape)
        # TODO: create attribute for 'center' & load accordingly if needed (normalization)

    def __len__(self):
        return len(self.sticks)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sticks[idx]).float()

    def statistics(self):
        mean = self.sticks.mean(0)
        std = self.sticks.std(0)
        return mean, std


def load_dataset(name):
    sticks = []
    for directory in tqdm(os.listdir('{}'.format(name))):
        directory = '{}/{}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:5] == 'DANCE':
            if os.path.exists(directory+'/skeletons.json'):
                with open(directory+'/skeletons.json') as f:
                    stick = json.load(f)
                    sticks.append(stick)
    return sticks


def stickwise(dataset):
    sticks = np.asarray(dataset[0]['skeletons'])
    for seq in tqdm(dataset[1:]):
        sticks = np.concatenate((sticks, np.asarray(seq['skeletons'])))
    return sticks


def visualize(pointsD, pointsG, title):
    plt.plot(list(range(0, pointsD.shape[0])), pointsD, label='lossD')
    plt.plot(list(range(0, pointsG.shape[0])), pointsG, label='lossG')
    plt.legend()
    plt.title(title)
    plt.show()

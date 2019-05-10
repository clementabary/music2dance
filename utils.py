import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset


class StickDataset(Dataset):
    def __init__(self, name):
        self.sticks = stickwise(load_dataset(name))
        # TODO: create attribute for 'center' & load accordingly if needed (normalization)

    def __len__(self):
        return len(self.sticks)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sticks[idx])


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

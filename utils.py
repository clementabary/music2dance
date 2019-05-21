import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class StickDataset(Dataset):
    def __init__(self, name, centering=True, normalize=None):
        sticks = load_dataset(name)
        self.skeletons = stickwise(sticks, 'skeletons')
        self.centers = stickwise(sticks, 'center')
        self.scaler = None
        if centering:
            self.skeletons = self.skeletons - self.centers[:, np.newaxis]
        if normalize == 'minmax':
            self.scaler = MinMaxScaler()
            dshape = np.shape(self.skeletons)
            self.skeletons = np.reshape(self.skeletons, (dshape[0], -1))
            self.skeletons = self.scaler.fit_transform(self.skeletons)
            self.skeletons = np.reshape(self.skeletons, dshape)
        # CHECK: is centering alright ? check statistics

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        return torch.from_numpy(self.skeletons[idx]).float()

    def statistics(self):
        mean = self.skeletons.mean(0)
        std = self.skeletons.std(0)
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


def stickwise(dataset, attribute):
    # attribute : 'skeletons', 'center'
    sticks = np.asarray(dataset[0][attribute])
    for seq in tqdm(dataset[1:]):
        sticks = np.concatenate((sticks, np.asarray(seq[attribute])))
    return sticks


def visualize(pointsD, pointsG, title):
    plt.plot(list(range(0, pointsD.shape[0])), pointsD, label='lossD')
    plt.plot(list(range(0, pointsG.shape[0])), pointsG, label='lossG')
    plt.legend()
    plt.title(title)
    plt.show()


def sampleG(model, noise=None):
    if noise is None:
        noise = torch.randn(1, model.latent_size)
        output = model(noise)
        example = output[0, :].detach().numpy()
        return np.reshape(example, (23, 3))
    else:
        outputs = model(noise)
        return outputs.detach().numpy()


def gradient_penalty(critic, bsize, real, fake):
    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)
    alpha = torch.rand(bsize, 1)
    alpha = alpha.expand(real.size())
    interpol = alpha * real + (1 - alpha) * fake
    interpol = interpol.view(interpol.size(0), 23, 3)
    interpol_critic = critic(interpol)
    gradients = autograd.grad(outputs=interpol_critic, inputs=interpol,
                              grad_outputs=torch.ones(interpol_critic.size()),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

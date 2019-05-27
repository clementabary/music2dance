import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.autograd as autograd
from sklearn.preprocessing import MinMaxScaler


class StickDataset(Dataset):
    def __init__(self, name, resume=False, centering=True, normalize=None):
        self.scaler = None
        if resume:
            self.skeletons = np.load(name)
        else:
            sticks = load_dataset(name)
            self.skeletons = stickwise(sticks, 'skeletons')
            self.centers = stickwise(sticks, 'center')
            if not centering:
                self.skeletons = self.skeletons + self.centers[:, np.newaxis]
        if normalize == 'minmax':
            self.scaler = MinMaxScaler()
            dshape = np.shape(self.skeletons)
            self.skeletons = np.reshape(self.skeletons, (dshape[0], -1))
            self.skeletons = self.scaler.fit_transform(self.skeletons)
            self.skeletons = np.reshape(self.skeletons, dshape)

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        return torch.from_numpy(self.skeletons[idx]).float()

    def statistics(self):
        mean = self.skeletons.mean(0)
        std = self.skeletons.std(0)
        return mean, std

    def export(self, path):
        np.save(path, self.skeletons)


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


def sampleG(model, noise=None):
    model.eval()
    if noise is None:
        noise = torch.randn(1, model.latent_size)
        output = model(noise)
        example = output[0, :].detach().numpy()
        return np.reshape(example, (23, 3))
    else:
        outputs = model(noise)
        return outputs.detach().cpu().numpy()


def gradient_penalty(critic, bsize, real, fake, device=None):
    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)
    alpha = torch.rand(bsize, 1)
    if device:
        alpha = alpha.expand(real.size()).to(device)
    else:
        alpha = alpha.expand(real.size())
    interpol = alpha * real.detach() + (1 - alpha) * fake.detach()
    interpol = interpol.view(interpol.size(0), 23, 3)
    interpol.requires_grad_(True)
    interpol_critic = critic(interpol)
    gradients = autograd.grad(outputs=interpol_critic, inputs=interpol,
                              grad_outputs=torch.ones(interpol_critic.size(), device=device),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

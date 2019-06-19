import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import librosa
import pickle
import sys


class StickDataset(Dataset):
    def __init__(self, name, resume=False, centering=True, normalize=None):
        self.scaler = None
        if resume:
            self.skeletons = np.load(name)
        else:
            sticks = load_sticks(name)
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


class SequenceDataset(Dataset):
    # dance_types arg for test purposes
    def __init__(self, name, resume=False, scaler=None, dance_types=['W', 'C', 'R', 'T']):
        self.scaler = None
        if resume:
            with open(name, 'rb') as f:
                dict = pickle.load(f)
                self.sequences = dict['sequences']
                self.labels = dict['labels']
                self.dirs = dict['dirs']
        else:
            sticks, musics, labels, dirs = load_all(name, dance_types)
            self.labels = labels
            self.dirs = dirs
            # self.musics = musics
            self.sequences = []
            # TODO: mix-max scaler for all sequences
            for _ in range(len(sticks)):
                self.sequences.append(np.asarray(sticks[_]['skeletons']))
        if scaler is not None:
            self.scaler = scaler
            for i, seq in enumerate(self.sequences):
                dshape = np.shape(seq)
                seq = np.reshape(seq, (seq.shape[0], -1))
                seq = self.scaler.transform(seq)
                self.sequences[i] = np.reshape(seq, dshape)

    def __len__(self):
        return (len(self.sequences))

    def __getitem__(self, idx):
        s, e = get_positions(self.sequences[idx])
        return (torch.from_numpy(self.sequences[idx][s:e]),
                # torch.from_numpy(self.musics[idx]),
                self.labels[idx], self.dirs[idx])

    def export(self, pathfile):
        dict = {'sequences': self.sequences,
                'labels': self.labels, 'dirs': self.dirs}
        with open(pathfile, 'wb') as f:
            pickle.dump(dict, f)
            f.close()


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    # sequences, musics, labels, dirs = zip(*batch)
    sequences, labels, dirs = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths), 23, 3)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    # return padded_seqs, lengths, musics, labels, dirs
    return padded_seqs, lengths, labels, dirs


def load_sticks(name):
    sticks = []
    for directory in tqdm(os.listdir('{}'.format(name))):
        directory = '{}/{}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:5] == 'DANCE':
            if os.path.exists(directory + '/skeletons.json'):
                with open(directory + '/skeletons.json') as f:
                    stick = json.load(f)
                    sticks.append(stick)
    return sticks


def load_all(name, dance_types):
    fps = 25
    sticks = []
    musics = []
    labels = []
    dirs = []
    for directory in tqdm(os.listdir('{}'.format(name))):
        directory = '{}/{}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:5] == 'DANCE':
            if os.path.basename(directory)[6] in dance_types:
                dirs.append(directory)
                labels.append(os.path.basename(directory)[6])
                with open(directory + '/config.json') as f:
                    config = json.load(f)
                    start, end = config['start_position'], config['end_position']
                    music, _ = librosa.load(directory + '/audio.mp3', sr=None,
                                            offset=start / fps,
                                            duration=(end - start) / fps)
                    musics.append(music)
                with open(directory + '/skeletons.json') as f:
                    stick = json.load(f)
                    sticks.append(stick)
    return sticks, musics, labels, dirs


def stickwise(dataset, attribute):
    # attribute : 'skeletons', 'center'
    sticks = np.asarray(dataset[0][attribute])
    for seq in tqdm(dataset[1:]):
        sticks = np.concatenate((sticks, np.asarray(seq[attribute])))
    return sticks


def sampleG(model, noise=None, device='cpu'):
    model.eval()
    if noise is None:
        noise = torch.randn(1, model.latent_size, device=device)
        output = model(noise)
        example = output[0, :].detach().cpu().numpy()
        return np.reshape(example, (23, 3))
    else:
        outputs = model(noise)
        return outputs.detach().cpu().numpy()


def sampleseqG(model, seq_length, noise=None, device='cpu'):
    model.eval()
    if noise is None:
        noise = torch.randn(
            1, seq_length, model.decoder.latent_size, device=device)
        output = model(noise, [seq_length])
        example = output.detach().cpu().numpy()
        return np.reshape(example, (seq_length, 23, 3))
    else:
        outputs = model(noise, [seq_length] * noise.shape[0])
        outputs = outputs.detach().cpu().numpy()
        return np.reshape(outputs, (seq_length * noise.shape[0], 23, 3))


def get_positions(sequence, min_s=120, max_s=300):
    l = len(sequence)
    s = np.random.randint(0, l - min_s)
    d = np.random.randint(min_s, min(l - s, max_s))
    return s, s + d


def extract_at_random(sequence, length):
    l = sequence.shape[1]
    idx = np.random.randint(0, l - length)
    return sequence[:, idx:idx + length]


def interpolate(input, fi):
    # choose new_len
    # delta = (len(inp)-1) / float(new_len-1)
    # output = [interpolate(inp, i*delta) for i in range(new_len)]
    i = int(fi)
    f = fi - i
    return (input[i] if f < sys.float_info.epsilon else
            input[i] + f * (input[i + 1] - input[i]))


def initialize_weights(net, initialisation=None, bias=None):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if initialisation is None:
                nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])
            if bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if initialisation is None:
                nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])
            if bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            if initialisation is None:
                nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])
            if bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            if initialisation is None:
                nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])
            if bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            if initialisation is None:
                nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])
            if bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GRU):
            for layer_params in m._all_weights:
                for param in layer_params:
                    if 'weight' in param:
                        if initialisation is None:
                            nn.init.xavier_normal_(m._parameters[param])
                        else:
                            nn.init.normal_(m._parameters[param],
                                            initialisation[0],
                                            initialisation[1])


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

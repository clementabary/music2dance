import torch.nn as nn
from utils import initialize_weights


class RecurrentDanceClassifier(nn.Module):
    # FIXME: handle variable size for training
    def __init__(self, channels_in, channels_h, output_code,
                 init_ker=9, n_blocks=1):
        super(RecurrentDanceClassifier, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_h,
                               kernel_size=init_ker,
                               padding=int((init_ker-1)/2))
        self.relu = nn.ReLU(inplace=True)
        self.blocks = []
        for _ in range(n_blocks):
            self.blocks.append(TemporalBlock(channels_h, 7))
        self.blocks = nn.Sequential(*self.blocks)
        self.rnn = nn.GRU(channels_h, output_code, batch_first=True)
        initialize_weights(self)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.blocks(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.rnn(x)[1]
        return x.squeeze(0)


class SimpleClassifier(nn.Module):
    def __init__(self, channels_in, channels_h, output_code):
        super(SimpleClassifier, self).__init__()
        self.rnn1 = nn.GRU(channels_in, channels_h, batch_first=True)
        self.rnn2 = nn.GRU(channels_h, output_code, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.rnn1(x)[0]
        x = self.rnn2(x)[1]
        return x.squeeze(0)


class DanceClassifier(nn.Module):
    def __init__(self, channels_in, channels_h, output_code,
                 seqlen, init_ker=9, n_blocks=2):
        super(DanceClassifier, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_h,
                               kernel_size=init_ker,
                               padding=int((init_ker-1)/2))
        self.blocks = []
        for _ in range(n_blocks):
            self.blocks.append(TemporalBlock(channels_h, 7))
        self.blocks = nn.Sequential(*self.blocks)
        self.fconv = nn.Conv1d(channels_h, output_code, seqlen)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.blocks(x)
        x = self.fconv(x)
        return x.squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(TemporalBlock, self).__init__()
        self.channels = channels
        self.ksize = ksize
        self.pad = int((self.ksize-1)/2)
        self.conv1 = nn.Conv1d(self.channels, self.channels,
                               kernel_size=self.ksize, padding=self.pad, dilation=1)
        self.conv2 = nn.Conv1d(self.channels, self.channels,
                               kernel_size=self.ksize, padding=self.pad, dilation=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_prime = self.relu(self.conv1(x))
        x_prime = self.relu(self.conv2(x_prime))
        return x + x_prime

import torch.nn as nn
from utils import initialize_weights


class SequenceGenerator(nn.Module):
    def __init__(self, input_size, latent_size, size, output_size, n_blocks, n_cells=1):
        super(SequenceGenerator, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.size = size
        self.output_size = output_size
        self.noise_gen = NoiseGen(input_size, latent_size, n_cells)
        self.decoder = FrameDecoder(latent_size, size, output_size, n_blocks)
        initialize_weights(self)

    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x = self.noise_gen(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.contiguous().view(-1, self.decoder.latent_size)
        outs = self.decoder(x)
        return outs


class SequenceDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, seqlen, n_blocks=1):
        super(SequenceDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_h, kernel_size=7, padding=3)
        self.blocks = []
        for _ in range(n_blocks):
            self.blocks.append(TemporalBlock(channels_h, 7))
        self.blocks = nn.Sequential(*self.blocks)
        self.lastconv = nn.Conv1d(channels_h, 1, seqlen)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        initialize_weights(self)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.blocks(x)
        x = self.lastconv(self.dropout(x))
        return x.squeeze(1)


class DiscriminatorAlpha(nn.Module):
    def __init__(self, channels_in):
        super(DiscriminatorAlpha, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, 128, 16, stride=2)
        self.conv2 = nn.Conv1d(128, 256, 4, stride=2)
        self.conv3 = nn.Conv1d(256, 512, 4, stride=2)
        self.maxpool1d = nn.MaxPool1d(4, stride=1)
        self.conv4 = nn.Conv1d(512, 1, 8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1d(self.conv3(x))
        x = self.conv4(x)
        return x.squeeze(1)


class DiscriminatorBeta(nn.Module):
    # use of more channels, wavenet-style dilation & dropout
    def __init__(self, channels_in):
        super(DiscriminatorBeta, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, 512, 32, stride=2, dilation=1)
        self.conv2 = nn.Conv1d(512, 512, 8, stride=2, dilation=2)
        self.conv3 = nn.Conv1d(512, 512, 4, stride=2, dilation=4)
        self.conv4 = nn.Conv1d(512, 1, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(self.dropout(x))
        return x.squeeze(1)


class NoiseGen(nn.Module):
    def __init__(self, input_size, output_size, n_layers):
        super(NoiseGen, self).__init__()
        self.rnn = nn.GRU(input_size, output_size, n_layers, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class FrameDecoder(nn.Module):
    def __init__(self, latent_size, size, output_size, nblocks):
        super(FrameDecoder, self).__init__()
        self.latent_size = latent_size
        self.size = size
        self.output_size = output_size
        self.nblocks = nblocks
        self.fc1 = nn.Linear(self.latent_size, self.size)
        self.bn1 = nn.BatchNorm1d(self.size, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = []
        for _ in range(self.nblocks):
            self.blocks.append(LinearBlock(self.size, use_bn=True))
        self.blocks = nn.Sequential(*self.blocks)
        self.dropout = nn.Dropout(p=0.5)
        self.lastfc = nn.Linear(self.size, self.output_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.blocks(x)
        x = self.lastfc(self.dropout(x))
        return x


class LinearBlock(nn.Module):
    def __init__(self, size, use_bn=False):
        super(LinearBlock, self).__init__()
        self.use_bn = use_bn
        self.size = size
        self.fc1 = nn.Linear(self.size, self.size, bias=True)
        self.fc2 = nn.Linear(self.size, self.size, bias=True)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(self.size, eps=1e-5, momentum=0.1)
            self.bn2 = nn.BatchNorm1d(self.size, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_prime = self.fc1(x)
        if self.use_bn:
            x_prime = self.bn1(x_prime)
        x_prime = self.relu(x_prime)
        x_prime = self.fc2(x)
        if self.use_bn:
            x_prime = self.bn2(x_prime)
        x_prime = self.relu(x_prime)
        return x + x_prime


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

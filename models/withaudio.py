import torch.nn as nn
import torch
from utils import initialize_weights


class SequenceGenerator(nn.Module):
    def __init__(self, window_size, input_size, latent_size,
                 size, output_size, n_blocks, n_cells=1):
        super(SequenceGenerator, self).__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.size = size
        self.output_size = output_size
        self.audio_enc = AudioEncoder(16, input_size)
        self.audio_rnn = NoiseGen(input_size, latent_size - 5, n_cells)
        self.noise_gen = NoiseGen(5, 5, 1)
        self.decoder = FrameDecoder(latent_size, size, output_size, n_blocks)
        initialize_weights(self)

    def forward(self, x, lengths, noise=None):
        # input shape (batch_size, seq_length, window_size)
        seq_length = x.size(1)
        x = x.view(-1, 1, self.window_size)
        x = self.audio_enc(x)
        x = x.view(-1, seq_length, self.input_size)
        if noise is None:
            s = list(x.size()[:-1])
            s.append(5)
            noise = torch.randn(s)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x = self.audio_rnn(x)
        n = self.noise_gen(noise)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = torch.cat((x, n), -1)
        x = x.contiguous().view(-1, self.decoder.latent_size)
        outs = self.decoder(x)
        return outs


class SequenceDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, output_code, seqlen):
        super(SequenceDiscriminator, self).__init__()
        self.stick_d = StickDiscriminator(channels_in, channels_h, output_code, seqlen)
        self.audio_d = AudioDiscriminator(output_code)
        self.fc1 = nn.Linear(2*output_code, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)
        initialize_weights(self)

    def forward(self, x, c):
        x = self.stick_d(x)
        c = self.audio_d(c)
        x = torch.cat((x, c), -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x


class AudioDiscriminator(nn.Module):
    def __init__(self, output_size):
        super(AudioDiscriminator, self).__init__()
        pad = 11
        self.l1 = nn.Conv1d(1, 16, 25, stride=4, padding=pad)
        self.l2 = nn.Conv1d(16, 32, 25, stride=4, padding=pad)
        self.l3 = nn.Conv1d(32, 64, 25, stride=4, padding=pad)
        self.l4 = nn.Conv1d(64, 128, 25, stride=4, padding=pad)
        self.l5 = nn.Conv1d(128, 256, 25, stride=4, padding=pad)
        self.l6 = nn.Conv1d(256, output_size, 75)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        x = self.l6(x)
        return x.squeeze(-1)


class StickDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, output_code, seqlen, init_ker=7, n_blocks=1):
        super(StickDiscriminator, self).__init__()
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
        x = self.relu(self. maxpool1d(self.conv3(x)))
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


class AudioEncoder(nn.Module):
    def __init__(self, f_maps, output_size):
        super(AudioEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, f_maps, 250, 50, 124))
        self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps), nn.ReLU(True)))
        for _ in range(5):
            self.conv_layers.append(nn.Conv1d(f_maps, f_maps*2, 4, 2, 1))
            self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps*2), nn.ReLU(True)))
            f_maps *= 2
        self.conv_layers.append(nn.Conv1d(512, output_size, 2))
        self.activations.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.activations)):
            x = self.conv_layers[i](x)
            x = self.activations[i](x)
        return x.squeeze()


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

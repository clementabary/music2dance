import torch.nn as nn
import torch
from utils import initialize_weights


class SequenceGenerator(nn.Module):
    def __init__(self, window_size, input_size, latent_size,
                 size, output_size, noise_size, n_blocks,
                 n_cells=1, enc_type="default", activ='id', device="cpu"):
        super(SequenceGenerator, self).__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.size = size
        self.noise_size = noise_size
        self.output_size = output_size
        self.device = device
        self.audio_enc = AudioEncoder(enc_type, 32, input_size, activ)
        self.audio_rnn = NoiseGen(input_size, latent_size - noise_size, n_cells)
        self.noise_gen = NoiseGen(noise_size, noise_size, 1)
        self.decoder = FrameDecoder(latent_size, size, output_size, n_blocks)
        initialize_weights(self)
        self.to(device)

    def forward(self, x, lengths, noise=None):
        # input shape (batch_size, seq_length, window_size)
        seq_length = x.size(1)
        x = x.view(-1, 1, self.window_size)
        x = self.audio_enc(x)
        x = x.view(-1, seq_length, self.input_size)
        if noise is None:
            s = list(x.size()[:-1])
            s.append(self.noise_size)
            noise = torch.randn(s).to(self.device)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x = self.audio_rnn(x)
        n = self.noise_gen(noise)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = torch.cat((x, n), -1)
        x = x.contiguous().view(-1, self.decoder.latent_size)
        outs = self.decoder(x)
        return outs


class AudioEncoder(nn.Module):
    def __init__(self, type, f_maps, output_size, activ='id'):
        super(AudioEncoder, self).__init__()
        if type == "default":
            self.model = DefaultAudioEncoder(f_maps, output_size, activ)
        elif type == "unet":
            self.model = UNetAudioEncoder(f_maps, output_size, activ)
        elif type == "wavegan":
            self.model = WaveGANAudioEncoder(f_maps, output_size, activ)

    def forward(self, x):
        return self.model(x)


class DefaultAudioEncoder(nn.Module):
    def __init__(self, f_maps, output_size, activ='id'):
        super(DefaultAudioEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, f_maps, 250, 50, 124))
        self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps), nn.ReLU(True)))
        for _ in range(5):
            self.conv_layers.append(nn.Conv1d(f_maps, f_maps*2, 4, 2, 1))
            self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps*2), nn.ReLU(True)))
            f_maps *= 2
        self.conv_layers.append(nn.Conv1d(f_maps, output_size, 2))
        if activ == 'id':
            self.activations.append(nn.Identity())
        elif activ == 'relu':
            self.activations.append(nn.ReLU(True))
        elif activ == 'tanh':
            self.activations.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.activations)):
            x = self.conv_layers[i](x)
            x = self.activations[i](x)
        return x.squeeze()


class UNetAudioEncoder(nn.Module):
    def __init__(self, f_maps, output_size, activ='id'):
        super(UNetAudioEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, f_maps, 160, 4, 79))
        self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps), nn.LeakyReLU(0.2)))
        for _ in range(2):
            self.conv_layers.append(nn.Conv1d(f_maps, f_maps*2, 4, 2, 1))
            self.activations.append(nn.Sequential(nn.BatchNorm1d(f_maps*2), nn.LeakyReLU(0.2)))
            f_maps *= 2
        self.ublock = UBlock(f_maps)
        self.fc = nn.Conv1d(f_maps, output_size, 200)
        if activ == 'id':
            self.activ = nn.Identity()
        elif activ == 'relu':
            self.activ = nn.ReLU(True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.activations)):
            x = self.conv_layers[i](x)
            x = self.activations[i](x)
        x = self.ublock(x)
        x = self.activ(self.fc(x))
        return x.squeeze()


class WaveGANAudioEncoder(nn.Module):
    def __init__(self, f_maps, output_size, activ='id'):
        super(WaveGANAudioEncoder, self).__init__()
        self.l1 = nn.Conv1d(1, f_maps, 25, stride=4)
        self.bn1 = nn.BatchNorm1d(f_maps)
        self.l2 = nn.Conv1d(f_maps, f_maps*2, 25, stride=4)
        f_maps *= 2
        self.bn2 = nn.BatchNorm1d(f_maps)
        self.l3 = nn.Conv1d(f_maps, f_maps*2, 25, stride=4)
        f_maps *= 2
        self.bn3 = nn.BatchNorm1d(f_maps)
        self.l4 = nn.Conv1d(f_maps, f_maps*2, 25, stride=4)
        f_maps *= 2
        self.bn4 = nn.BatchNorm1d(f_maps)
        self.l5 = nn.Conv1d(f_maps, output_size, 5)
        self.relu = nn.ReLU(True)
        if activ == 'id':
            self.activ = nn.Identity()
        elif activ == 'relu':
            self.activ = nn.ReLU(True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))
        x = self.relu(self.bn3(self.l3(x)))
        x = self.relu(self.bn4(self.l4(x)))
        x = self.activ(self.l5(x))
        return x.squeeze(-1)


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
        # self.dropout = nn.Dropout(p=0.5)
        self.lastfc = nn.Linear(self.size, self.output_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.blocks(x)
        # x = self.lastfc(self.dropout(x))
        x = self.lastfc(x)
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


class BasisConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(BasisConvBlock, self).__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, 3, 1, 1)
        self.bn = nn.BatchNorm1d(channels_out)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UBlock(nn.Module):
    def __init__(self, channels):
        super(UBlock, self).__init__()
        self.convblock1 = BasisConvBlock(channels, channels)
        self.convblock2 = BasisConvBlock(channels, channels)
        self.convblock3 = BasisConvBlock(channels, channels)
        self.convblock4 = BasisConvBlock(channels, channels)
        self.convblock5 = BasisConvBlock(channels*2, channels)
        self.convblock6 = BasisConvBlock(channels*2, channels)
        self.convblock7 = BasisConvBlock(channels*2, channels)

        self.downsample = nn.MaxPool1d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(self.downsample(x1))
        x3 = self.convblock3(self.downsample(x2))
        x4 = self.convblock4(self.downsample(x3))
        x3 = self.convblock5(torch.cat((self.upsample(x4), x3), 1))
        x2 = self.convblock6(torch.cat((self.upsample(x3), x2), 1))
        x = self.convblock7(torch.cat((self.upsample(x2), x1), 1))
        return x


class SequenceDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, output_code, seqlen,
                 init_ker=9, activ='id', device="cpu"):
        super(SequenceDiscriminator, self).__init__()
        self.stick_d = StickDiscriminator(channels_in, channels_h, output_code,
                                          seqlen, init_ker=init_ker, activ=activ)
        self.audio_d = AudioDiscriminator(output_code, activ)
        self.fc1 = nn.Linear(2*output_code, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(True)
        #  self.dropout = nn.Dropout(0.2)
        initialize_weights(self)
        self.to(device)

    def forward(self, x, c):
        x = self.stick_d(x)
        c = self.audio_d(c)
        x = torch.cat((x, c), -1)
        # x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AblatedSequenceDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, output_code, seqlen,
                 init_ker=9, activ='id', device="cpu"):
        super(AblatedSequenceDiscriminator, self).__init__()
        self.stick_d = StickDiscriminator(channels_in, channels_h, output_code,
                                          seqlen, activ=activ)
        self.fc1 = nn.Linear(output_code, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(True)
        #  self.dropout = nn.Dropout(0.2)
        initialize_weights(self)
        self.to(device)

    def forward(self, x):
        x = self.stick_d(x)
        # x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AudioDiscriminator(nn.Module):
    def __init__(self, output_size, activ='id'):
        super(AudioDiscriminator, self).__init__()
        pad = 11
        self.l1 = nn.Conv1d(1, 32, 25, stride=4, padding=pad)
        self.l2 = nn.Conv1d(32, 64, 25, stride=4, padding=pad)
        self.l3 = nn.Conv1d(64, 128, 25, stride=4, padding=pad)
        self.l4 = nn.Conv1d(128, 256, 25, stride=4, padding=pad)
        self.l5 = nn.Conv1d(256, 512, 25, stride=4, padding=pad)
        self.l6 = nn.Conv1d(512, output_size, 75)
        self.relu = nn.ReLU(True)
        if activ == 'id':
            self.activ = nn.Identity()
        elif activ == 'relu':
            self.activ = nn.ReLU(True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        x = self.activ(self.l6(x))
        return x.squeeze(-1)


class StickDiscriminator(nn.Module):
    def __init__(self, channels_in, channels_h, output_code, seqlen,
                 init_ker=9, n_blocks=2, activ='id'):
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
        if activ == 'id':
            self.activ = nn.Identity()
        elif activ == 'relu':
            self.activ = nn.ReLU(True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.blocks(x)
        x = self.activ(self.fconv(x))
        return x.squeeze(-1)


class NoiseGen(nn.Module):
    def __init__(self, input_size, output_size, n_layers):
        super(NoiseGen, self).__init__()
        self.rnn = nn.GRU(input_size, output_size, n_layers, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]

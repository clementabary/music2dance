import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, size, output_size, nblocks):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.nblocks = nblocks
        self.fc1 = nn.Linear(latent_size, size)
        self.bn1 = nn.BatchNorm1d(size, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = []
        for _ in range(self.nblocks):
            self.blocks.append(LinearBlock(size, use_bn=True))
        self.blocks = nn.Sequential(*self.blocks)
        self.dropout = nn.Dropout(p=0.5)
        self.lastfc = nn.Linear(size, output_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.blocks(x)
        x = self.lastfc(self.dropout(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, size, nblocks):
        super(Discriminator, self).__init__()
        self.nblocks = nblocks
        self.fc1 = nn.Linear(input_size, size)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = []
        for _ in range(self.nblocks):
            self.blocks.append(LinearBlock(size))
        self.blocks = nn.Sequential(*self.blocks)
        self.dropout = nn.Dropout(p=0.5)
        self.lastfc = nn.Linear(size, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.blocks(x)
        x = self.lastfc(self.dropout(x))
        return x


class LinearBlock(nn.Module):
    def __init__(self, size, use_bn=False):
        super(LinearBlock, self).__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(size, size, bias=True)
        self.fc2 = nn.Linear(size, size, bias=True)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(size, eps=1e-5, momentum=0.1)
            self.bn2 = nn.BatchNorm1d(size, eps=1e-5, momentum=0.1)
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

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 69)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))
        out = self.sigmoid(self.fc3(x))
        return out.squeeze(1)

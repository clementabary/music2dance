from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.mlp import Generator, Discriminator


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(datasetf)

# TODO: add normalization transforms

# Training hyper-parameters
num_train = 10000  # 70000
num_epochs = 10
batch_size = 8

# Loading dataset
writer = SummaryWriter()
dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(range(num_train)))
n_iter = len(dataloader)

# Creating models
latent_vector_size = 10
real_label = torch.full((batch_size,), 1)
fake_label = torch.full((batch_size,), 0)
fixed_noise = torch.randn(batch_size, latent_vector_size)
criterion = nn.BCELoss(reduction='mean')

modelG = Generator(latent_vector_size)
sum(p.numel() for p in modelG.parameters() if p.requires_grad)

modelD = Discriminator(69)
sum(p.numel() for p in modelD.parameters() if p.requires_grad)

optimizerD = optim.Adam(modelD.parameters(), lr=0.0002)
optimizerG = optim.Adam(modelG.parameters(), lr=0.0002)

lossesD = []
lossesG = []

# Training process : vanilla GAN
for epoch in range(num_epochs):
    for idx, real in enumerate(dataloader):

        # train discriminator modelD
        lossD = 0
        optimizerD.zero_grad()
        output = modelD(real)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        noise = torch.randn(batch_size, latent_vector_size)
        fake = modelG(noise)
        output = modelD(fake.detach())
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        lossD += errD.item()
        optimizerD.step()

        # train generator modelG
        lossG = 0
        optimizerG.zero_grad()
        output = modelD(fake)
        errG = criterion(output, real_label)
        errG.backward()
        lossG += errG.item()
        optimizerG.step()

        writer.add_scalar('lossD', lossD, idx + epoch * n_iter)
        writer.add_scalar('lossG', lossG, idx + epoch * n_iter)

        for name, param in modelG.named_parameters():
            writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)
        for name, param in modelG.named_parameters():
            writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)

    lossesD.append(lossD)
    lossesG.append(lossG)
    print('Epoch {}/{} :  LossD: {} LossG: {}'.format(epoch+1, num_epochs, lossD, lossG))

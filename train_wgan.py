from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset
from utils import sampleG
import torch
import torch.optim as optim
import numpy as np
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.mlp import Generator, Discriminator
from visualize import to_2d_graph_data, visualize_2d_graph


# TODO: check normalization techniques, architectures (layers)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if not os.path.exists('./samples'):
    os.makedirs('./samples')

# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(datasetf, centering=True, normalize='minmax')

# Training hyper-parameters
num_train = 5000  # 70000
num_epochs = 10
batch_size = 8
nb_samples = 5

# Loading dataset
logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_wgan'
writer = SummaryWriter('./runs/' + logdir)
os.makedirs('./samples/' + logdir)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(range(num_train)))
n_iter = len(dataloader)
n_critic_steps = 5
threshold = 0.1

# Creating models
latent_vector_size = 50
real_label = torch.full((batch_size,), 1)
fake_label = torch.full((batch_size,), 0)
fixed_noise = torch.randn(nb_samples, latent_vector_size)

gen = Generator(latent_vector_size)
sum(p.numel() for p in gen.parameters() if p.requires_grad)

critic = Discriminator(69)
sum(p.numel() for p in critic.parameters() if p.requires_grad)

optim_critic = optim.Adam(critic.parameters(), lr=0.0002)
optim_gen = optim.Adam(gen.parameters(), lr=0.0002)

losses_critic = []
losses_gen = []

# Training process : Wasserstein GAN (with weight clipping)
for epoch in range(num_epochs):
    gen.train()
    for idx, real in enumerate(dataloader):

        loss_critic = 0
        for _ in range(n_critic_steps):
            # train discriminator critic n_critic_steps times
            optim_critic.zero_grad()
            err_real = torch.mean(critic(real))

            noise = torch.randn(batch_size, latent_vector_size)
            fake = gen(noise)
            err_fake = torch.mean(critic(fake.detach()))
            err_critic = err_fake - err_real
            loss_critic += err_critic.item()
            err_critic.backward()
            optim_critic.step()
            for p in critic.parameters():
                p.data.clamp_(-threshold, threshold)
        loss_critic /= n_critic_steps

        # train generator gen
        loss_gen = 0
        optim_gen.zero_grad()
        noise = torch.randn(batch_size, latent_vector_size)
        fake = gen(noise)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake))
        err_gen = err_real - err_fake
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', loss_critic, idx + epoch * n_iter)
        writer.add_scalar('loss_gen', loss_gen, idx + epoch * n_iter)

        # for name, param in gen.named_parameters():
        #     writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)
        # for name, param in gen.named_parameters():
        #     writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)

    losses_critic.append(loss_critic)
    losses_gen.append(loss_gen)
    print('Epoch {}/{} : loss_critic: {} loss_gen: {}'.format(epoch+1, num_epochs,
                                                              loss_critic, loss_gen))

    if (epoch + 1) % 5 == 0:
        gen.eval()
        print("Generating samples...")
        samples = sampleG(gen, fixed_noise)
        if dataset.scaler is not None:
            samples = dataset.scaler.inverse_transform(samples)
        for s in range(nb_samples):
            trace_2d = to_2d_graph_data(np.reshape(samples[s, :], (23, 3)))
            filepath = './samples/' + logdir + '/e{}s{}.png'.format(epoch+1, s+1)
            visualize_2d_graph(trace_2d, save=filepath)

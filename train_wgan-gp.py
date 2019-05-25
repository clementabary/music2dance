from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset
from utils import sampleG, gradient_penalty
import torch
import torch.optim as optim
import numpy as np
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.mlp import Generator, Discriminator  # models.haoye
from visualize import to_2d_graph_data, visualize_2d_graph
# from tensorboardcolab import *


GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if not os.path.exists('./samples'):
    os.makedirs('./samples')

if not os.path.exists('./models'):
    os.makedirs('./models')

# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(datasetf, centering=True, normalize='minmax')

# Training hyper-parameters
num_train = 64000  # 70000
num_epochs = 20
batch_size = 64
nb_samples = 5
gamma = 0.1
nblocks = 1
latent_vector_size = 100
size = 512
output_size = 69
lr = 0.0002
n_critic_steps = 5

# Loading dataset
logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_wgan-gp'
writer = SummaryWriter('./runs/' + logdir)
# tbc = tensorboardcolab.TensorBoardColab(graph_path='./runs/'+logdir)  # for Colab
os.makedirs('./samples/' + logdir)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(range(num_train)))
n_iter = len(dataloader)

# Creating models
real_label = torch.full((batch_size,), 1, device=device)
fake_label = torch.full((batch_size,), 0, device=device)
fixed_noise = torch.randn(nb_samples, latent_vector_size, device=device)

gen = Generator(latent_vector_size, size, output_size, nblocks)
sum(p.numel() for p in gen.parameters() if p.requires_grad)
gen.to(device)

critic = Discriminator(output_size, size, nblocks)
sum(p.numel() for p in critic.parameters() if p.requires_grad)
critic.to(device)

optim_critic = optim.Adam(critic.parameters(), lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)

losses_critic = []
losses_gen = []

# Training process : Wasserstein GAN-GP
for epoch in range(num_epochs):
    gen.train()
    for idx, real in enumerate(dataloader):

        loss_critic = 0
        for _ in range(n_critic_steps):
            # Train discriminator critic n_critic_steps times
            optim_critic.zero_grad()
            noise = torch.randn(batch_size, latent_vector_size, device=device)
            fake = gen(noise)
            gp = gradient_penalty(critic, batch_size, real, fake, device=device)
            err_real = torch.mean(critic(real))
            err_fake = torch.mean(critic(fake.detach()))
            err_critic = err_fake - err_real + gamma * gp
            loss_critic += err_critic.item()
            err_critic.backward()
            optim_critic.step()
        loss_critic /= n_critic_steps

        # Train generator gen
        loss_gen = 0
        optim_gen.zero_grad()
        noise = torch.randn(batch_size, latent_vector_size, device=device)
        fake = gen(noise)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake))
        err_gen = err_real - err_fake
        loss_gen += err_gen.item()
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', loss_critic, idx + epoch * n_iter)
        writer.add_scalar('loss_gen', loss_gen, idx + epoch * n_iter)

        # for name, param in gen.named_parameters():
        #     writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)
        # for name, param in gen.named_parameters():
        #     writer.add_histogram(name, param.data.numpy(), idx + epoch * n_iter)

        # tbc.save_value('Loss D', 'lossD', idx + epoch * n_iter, loss_critic)  # Colab
        # tbc.save_value('Loss G', 'lossG', idx + epoch * n_iter, loss_gen)

    losses_critic.append(loss_critic)
    losses_gen.append(loss_gen)
    print('Epoch {}/{} : loss_critic: {} loss_gen: {}'.format(epoch+1, num_epochs,
                                                              loss_critic, loss_gen))

    # Generate visualizations with fixed noise every few epochs
    if (epoch + 1) % 5 == 0:
        print("Generating samples...")
        samples = sampleG(gen, fixed_noise)
        if dataset.scaler is not None:
            samples = dataset.scaler.inverse_transform(samples)
        for s in range(nb_samples):
            trace_2d = to_2d_graph_data(np.reshape(samples[s, :], (23, 3)))
            filepath = './samples/' + logdir + '/e{}s{}.png'.format(epoch+1, s+1)
            visualize_2d_graph(trace_2d, save=filepath)
            # filepath = './samples/' + logdir + '/e{}s{}.html'.format(epoch+1, s+1)
            # graph_fig_2d = visualize_2d_graph(trace_2d)  # Colab
            # plot(graph_fig_2d, filename=filepath, auto_open=False)

torch.save(gen.state_dict(), './models/' + logdir + '_gen_{}.pt'.format(num_epochs))
torch.save(critic.state_dict(), './models/' + logdir + '_critic_{}.pt'.format(num_epochs))

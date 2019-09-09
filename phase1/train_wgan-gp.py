from phase1.archis.residual import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from music2dance.visualize import to_2d_graph_data, visualize_2d_graph
from utils import StickDataset
from losses import gradient_penalty
import torch
import torch.optim as optim
import numpy as np
import os
import datetime
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=int, help="choose gpu id")
parser.add_argument("-n", "--name", type=str, help="choose name of experiment")
opts = parser.parse_args()

device_idx = opts.device
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

print("Loading sticks and sequences datasets...")
data_folder = '../../Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(data_folder, normalize='minmax')

if not os.path.exists('./runs'):
    os.makedirs('./runs')

with open(file, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + opts.name
os.makedirs('./runs/' + logdir)

# Training hyper-parameters
num_train = cfg['num_train']
num_epochs = cfg['num_epochs']
batch_size = cfg['batch_size']
gamma = cfg['gamma']
nb_samples = 5
nblocks_gen = cfg['nblocks_gen']
nblocks_critic = cfg['nblocks_critic']
latent_vector_size = cfg['latent_vector_size']
size = cfg['size']
output_size = cfg['output_size']
lr_gen = cfg['lr_gen']
lr_critic = cfg['lr_critic']
n_critic_steps = cfg['n_critic_steps']

random_seed = 37
np.random.seed(random_seed)
fixed_noise = torch.randn(nb_samples, latent_vector_size, device=device)

# Loading dataset
print('Loading models..')
gen = Generator(latent_vector_size, size, output_size, nblocks_gen).to(device)
critic = Discriminator(output_size, size, nblocks_critic).to(device)

optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)
optim_gen = optim.Adam(gen.parameters(), lr=lr_gen)

print('Prepare for logging..')
writer = SummaryWriter('./runs/' + logdir + '/logging')
sampledir = './runs/' + logdir + '/samples'
os.makedirs(sampledir)
modeldir = './runs/' + logdir + '/models'
os.makedirs(modeldir)

dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(range(num_train)))

# Training process : Wasserstein GAN-GP
print('Start training..')
total_iterations = 0
for epoch in range(num_epochs):
    gen.train()
    for idx, real in enumerate(dataloader):
        total_iterations += 1
        # Train discriminator critic n_critic_steps times
        optim_critic.zero_grad()
        real = real.to(device)
        noise = torch.randn(batch_size, latent_vector_size, device=device)
        fake = gen(noise)
        gp = gradient_penalty(critic, batch_size, real, fake, device=device)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake.detach()))
        err_critic = err_fake - err_real + gamma * gp
        w_dist = err_fake - err_real
        loss_critic = err_critic.item()
        err_critic.backward()
        optim_critic.step()

        if total_iterations % n_critic_steps:
            continue

        # Train generator gen
        optim_gen.zero_grad()
        noise = torch.randn(batch_size, latent_vector_size, device=device)
        fake = gen(noise)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake))
        err_gen = err_real - err_fake
        loss_gen = err_gen.item()
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', -loss_critic, total_iterations)
        writer.add_scalar('loss_gen', loss_gen, total_iterations)

    print('Epoch {}/{} : loss_critic: {} loss_gen: {}'.format(epoch+1, num_epochs,
                                                              loss_critic, loss_gen))

    # Generate visualizations with fixed noise every few epochs
    if (epoch + 1) % 5 == 0:
        # gen.eval()
        # print("Generating samples...")
        # samples = sampleG(gen, fixed_noise)
        # if dataset.scaler is not None:
        #     samples = dataset.scaler.inverse_transform(samples)
        # for s in range(nb_samples):
        #     trace_2d = to_2d_graph_data(np.reshape(samples[s, :], (23, 3)))
        #     filepath = sampledir + '/e{}s{}.png'.format(epoch+1, s+1)
        #     visualize_2d_graph(trace_2d, save=filepath)
        torch.save(gen.state_dict(), modeldir + '/gen_{}.pt'.format(epoch + 1))
        torch.save(critic.state_dict(), modeldir + '/critic_{}.pt'.format(epoch + 1))

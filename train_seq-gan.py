from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import SequenceDataset
from utils import gradient_penalty, collate_fn, extract_at_random
from utils import sampleseqG
import torch
import torch.optim as optim
import os
import numpy as np
import datetime
from models.temporal import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from visualize import frame_to_vid


GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx)
                          if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if not os.path.exists('./runs'):
    os.makedirs('./runs')

logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_seq-wgan-gp'
os.makedirs('./runs/' + logdir)

# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = SequenceDataset(datasetf)

# Training hyper-parameters
num_train = 50  # 61
num_epochs = 1000
batch_size = 4
nb_samples = 5
seq_length = 120
gamma = 0.1
nblocks = 2
latent_vector_size = 50
size = 512
output_size = 69
lr = 0.0002
n_critic_steps = 8

# Loading dataset
writer = SummaryWriter('./runs/' + logdir + '/logging')
sampledir = './runs/' + logdir + '/samples'
os.makedirs(sampledir)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        sampler=SubsetRandomSampler(range(num_train)))
n_iter = len(dataloader)

# Creating models
fixed_noise = torch.randn(nb_samples, seq_length,
                          latent_vector_size, device=device)

gen = Generator(latent_vector_size, size, output_size, nblocks)
sum(p.numel() for p in gen.parameters() if p.requires_grad)
gen.to(device)

critic = Discriminator(output_size)
sum(p.numel() for p in critic.parameters() if p.requires_grad)
critic.to(device)

optim_critic = optim.Adam(critic.parameters(), lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)

total_iterations = 0

# Training process : "Sequential" Wasserstein GAN-GP
for epoch in range(num_epochs):
    gen.train()
    for idx, (real, _, _, _) in enumerate(dataloader):
        total_iterations += 1

        # Train discriminator critic n_critic_steps times
        optim_critic.zero_grad()

        noise = torch.randn(batch_size, seq_length,
                            latent_vector_size, device=device)
        fake = gen(noise, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        real = real.to(device)
        real_s = extract_at_random(real, seq_length)
        real_s = real_s.view(batch_size, seq_length,
                             output_size).permute(0, 2, 1).contiguous()

        gp = gradient_penalty(critic, batch_size,
                              real_s, fake, is_seq=True, device=device)

        # Compute wasserstein distance
        err_real = torch.mean(critic(real_s))
        err_fake = torch.mean(critic(fake.detach()))
        err_critic = err_fake - err_real + gamma * gp
        loss_critic = err_critic.item()
        err_critic.backward(retain_graph=True)
        optim_critic.step()

        if total_iterations % n_critic_steps:
            continue

        # Train generator gen
        optim_gen.zero_grad()
        noise = torch.randn(batch_size, seq_length,
                            latent_vector_size, device=device)
        fake = gen(noise, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        err_real = torch.mean(critic(real_s))
        err_fake = torch.mean(critic(fake))
        err_gen = err_real - err_fake
        loss_gen = err_gen.item()
        err_gen.backward()
        optim_gen.step()

        # writer.add_scalar('loss_critic', loss_critic, total_iterations)
        # writer.add_scalar('loss_gen', loss_gen, total_iterations)

    if (epoch + 1) % 100 == 0:
        gen.eval()
        print("Generating sample animations...")
        samples = sampleseqG(gen, seq_length, fixed_noise)
        samples = np.reshape(samples, (seq_length * nb_samples, -1))
        if dataset.scaler is not None:
            samples = dataset.scaler.inverse_transform(samples)
        samples = np.reshape(samples, (nb_samples, seq_length, 23, 3))
        for s in range(nb_samples):
            filepath = sampledir + '/e{}s{}.avi'.format(epoch + 1, s + 1)
            frame_to_vid(samples[s, :], filepath, fps=25)

# torch.save(gen.state_dict(), './models/' + logdir + '_gen_{}.pt'.format(num_epochs))
# torch.save(critic.state_dict(), './models/' + logdir + '_critic_{}.pt'.format(num_epochs))

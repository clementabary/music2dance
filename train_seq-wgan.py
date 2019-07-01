from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset, SequenceDataset
from utils import collate_fn, sampleseqG  # , freeze
from losses import gradient_penalty, tv_loss
from visualize import frame_to_vid
import torch
import torch.optim as optim
import os
import datetime
import numpy as np
from models.tempconv import SequenceGenerator, SequenceDiscriminator
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import json

# for Colab notebook, run first :
# !pip install -q tb-nightly
# from tensorflow import summary
# %load_ext tensorboard

'''
WGAN-GP/LP framework for generating dance sequences from
Music-to-Dance-Motion-synthesis dataset

The generator is recurrent but the critic is 1D-temporal fully convolutional
Training on randomly sampled sequences of fixed length
'''

# Check for GPU access
GPU = True
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

# Preparing folders
if not os.path.exists('./runs'):
    os.makedirs('./runs')

logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_seq-wgan-gp'
os.makedirs('./runs/' + logdir)

# Load dataset
# Notice : flaws in waltz files : need for interpolation and re-sampling before
# StickDataset is used to get 'minmax' scaler and for test purposes
sticks = StickDataset('./data/fixed_centered_skeletons.npy',
                      resume=True,
                      normalize='minmax')
dataset = SequenceDataset('./data/fixed_seqds.pkl', resume=True, scaler=sticks.scaler)

# Training hyper-parameters
num_train = 60               # 61 in total
num_epochs = 100000          # approximately 2h on Colab
batch_size = 30              # FIXME: needs to divide num_train
nb_samples = 5               # number of samples generated during training
seq_length = 120             # fixed length of sequence for training
gamma = 10                   # gradient penalty coefficient
eta = 50                     # total variation penalty coefficient
nblocks_gen = 2              # number of residual blocks in generator
nblocks_critic = 3           # number of residual blocks in critic
input_vector_size = 250      # input size for GRU layer in generator
latent_vector_size = 100     # output size of GRU and input size for decoder
n_cells = 2                  # number of GRU layers in noise generator
size = 128                   # size of FC layers in decoder
channels = 256               # number of channels in convolutional discriminator
output_size = 69             # corresponds to 3*23 joints
lr_gen = 0.0005              # learning rate for generator
lr_critic = 0.0005           # learning rate for critic
n_critic_steps = 8           # times critic is updated before generator
freeze_epoch = False         # number of epochs before freezing decoder
init_kernel = 25             # kernel of first 1D convolution in critic
fixed_noise = torch.randn(nb_samples, seq_length,
                          input_vector_size, device=device)

# Creating models
gen = SequenceGenerator(input_vector_size, latent_vector_size,
                        size, output_size, nblocks_gen, n_cells)
print(sum(p.numel() for p in gen.parameters() if p.requires_grad))
gen.to(device)

critic = SequenceDiscriminator(
    output_size, channels, seq_length, init_kernel, nblocks_critic)
print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
critic.to(device)

# Creating optimizers and schedulers
optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)
optim_gen = optim.Adam(gen.parameters(), lr=lr_gen)
scheduler_critic = MultiStepLR(optim_critic, milestones=[10000, 35000, 50000], gamma=0.9)
scheduler_gen = MultiStepLR(optim_gen, milestones=[10000, 35000, 50000], gamma=0.9)

# Logging
writer = SummaryWriter('./runs/' + logdir + '/logging')
sampledir = './runs/' + logdir + '/samples'
os.makedirs(sampledir)

# Loading data
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        sampler=SubsetRandomSampler(range(num_train)))

# in Colab notebook :
# %tensorboard --logdir=runs

# Training process : "Sequential" Wasserstein GAN-GP/LP
total_iterations = 0
for epoch in range(num_epochs):
    gen.train()
    for (real, _, _, _) in dataloader:
        total_iterations += 1

        optim_critic.zero_grad()
        noise = torch.randn(batch_size, seq_length,
                            input_vector_size, device=device)
        fake = gen(noise, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        real = real.to(device)
        real = real.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        gp = gradient_penalty(critic, batch_size, real,
                              fake, is_seq=True, lp=True, device=device)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake.detach()))
        err_critic = err_fake - err_real + gamma * gp
        w_dist = err_fake - err_real
        loss_critic = err_critic.item()
        err_critic.backward(retain_graph=True)
        optim_critic.step()

        if total_iterations % n_critic_steps:
            continue

        # Train generator gen
        optim_gen.zero_grad()
        noise = torch.randn(batch_size, seq_length,
                            input_vector_size, device=device)
        fake = gen(noise, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length,
                         output_size).permute(0, 2, 1)
        err_real = torch.mean(critic(real))
        err_fake = torch.mean(critic(fake))
        tvp = tv_loss(fake)
        err_gen = err_real - err_fake + eta * tvp
        loss_gen = err_gen.item()
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', loss_critic, total_iterations)
        writer.add_scalar('loss_gen', loss_gen, total_iterations)
        writer.add_scalar('gp', gp, total_iterations)
        writer.add_scalar('w_dist', w_dist, total_iterations)

        scheduler_critic.step()
        scheduler_gen.step()

    # if (epoch + 1) > freeze_epoch:
    #     freeze(gen.decoder)

    if (epoch + 1) % 500 == 0:
        print("Iteration: {} LossG : {} LossD : {}".format(
            total_iterations, loss_gen, loss_critic))

    # Generating sample animations of 120 frames i.e. 4.8s
    if (epoch + 1) % 5000 == 0:
        gen.eval()
        print("Generating sample animations...")
        samples = sampleseqG(gen, seq_length, fixed_noise)
        samples = np.reshape(samples, (seq_length * nb_samples, -1))
        if dataset.scaler is not None:
            samples = dataset.scaler.inverse_transform(samples)
            samples = np.reshape(samples, (nb_samples, seq_length, 23, 3))
            for s in range(nb_samples):
                frame_to_vid(samples[s, :], sampledir +
                             '/e{}s{}.avi'.format(epoch + 1, s), fps=25)


# Saving weights
modeldir = './runs/' + logdir + '/models'
os.makedirs(modeldir)
torch.save(gen.state_dict(), modeldir + '/gpgen_{}.pt'.format(num_epochs))
torch.save(critic.state_dict(), modeldir + '/gpcritic_{}.pt'.format(num_epochs))

# Saving architectures
with open('./runs/' + logdir + '/model_gen.txt', 'w+') as f:
    f.write(str(gen))
    f.close()

with open('./runs/' + logdir + '/model_critic.txt', 'w+') as f:
    f.write(str(critic))
    f.close()

# Saving hyperparameters
config = {'num_train': num_train,
          'num_epochs': num_epochs,
          'batch_size': batch_size,
          'nb_samples': nb_samples,
          'seq_length': seq_length,
          'gamma': gamma,
          'nblocks_gen': nblocks_gen,
          'nblocks_critic': nblocks_critic,
          'input_vector_size': input_vector_size,
          'latent_vector_size': latent_vector_size,
          'size': size,
          'channels': channels,
          'output_size': output_size,
          'lr_gen': lr_gen,
          'lr_critic': lr_critic,
          'n_critic_steps': n_critic_steps,
          'n_cells': n_cells,
          'lp': True
          }

with open('./runs/' + logdir + '/config.json', 'w') as f:
    json.dump(config, f)

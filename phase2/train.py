from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset, SequenceDataset
from utils import sampleseqG, collate_fn
from visualize import frame_to_vid
from losses import gradient_penalty, tv_loss
import torch
import torch.optim as optim
import os
import datetime
import numpy as np
from phase2.models.tempconv import SequenceGenerator, SequenceDiscriminator
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml


# Check for GPU access
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose config file")
parser.add_argument("-d", "--device", type=int, help="choose gpu id")
parser.add_argument("-n", "--name", type=str, help="name experiment")
parser.add_argument("-f", "--framework", type=str)
opts = parser.parse_args()

device_idx = opts.device
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

with open(opts.config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Preparing folders
if not os.path.exists('./runs'):
    os.makedirs('./runs')

logdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + opts.name
os.makedirs('./runs/' + logdir)

# Training hyper-parameters
num_train = cfg['num_train']
num_epochs = cfg['num_epochs']
batch_size = cfg['batch_size']
seq_length = cfg['seq_length']
gamma = cfg['gamma']
eta = cfg['eta']
nblocks_gen = cfg['nblocks_gen']
nblocks_critic = cfg['nblocks_critic']
input_vector_size = cfg['input_vector_size']
latent_vector_size = cfg['latent_vector_size']
n_cells = cfg['n_cells']
size = cfg['size']
channels = cfg['channels']
output_size = cfg['output_size']
lr_gen = cfg['lr_gen']
lr_critic = cfg['lr_critic']
n_critic_steps = cfg['n_critic_steps']
freeze_epoch = cfg['freeze_epoch']
init_kernel = cfg['init_kernel']
nb_samples = 5

# Creating datasets
print("Loading sticks and sequences datasets...")
sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=False)

stick_length = dataset.stick_length
fixed_noise = torch.randn(nb_samples, stick_length, input_vector_size, device=device)

# Creating models
print("Loading models..")
gen = SequenceGenerator(input_vector_size, latent_vector_size,
                        size, output_size, nblocks_gen, n_cells, device)
# print(sum(p.numel() for p in gen.parameters() if p.requires_grad))

critic = SequenceDiscriminator(output_size, channels, dataset.stick_length,
                               init_kernel, nblocks_critic, device)
# print(sum(p.numel() for p in critic.parameters() if p.requires_grad))

# Creating optimizers and schedulers
optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)
optim_gen = optim.Adam(gen.parameters(), lr=lr_gen)
scheduler_critic = MultiStepLR(optim_critic, milestones=[10000, 35000, 50000], gamma=0.8)
scheduler_gen = MultiStepLR(optim_gen, milestones=[10000, 35000, 50000], gamma=0.8)

# Logging
print("Prepare for logging..")
writer = SummaryWriter('./runs/' + logdir + '/logging')
sampledir = './runs/' + logdir + '/samples'
os.makedirs(sampledir)
modeldir = './runs/' + logdir + '/models'
os.makedirs(modeldir)

# Loading data
# BE CAREFUL: check if len train/valid can be divided by batch_size !
print("Loading data..")  # 0.7, 0.2, 0.1
# train_indices = [101, 60, 58, 55, 50, 46, 7, 5, 67, 14, 72, 25, 91, 87,
#                  89, 1, 117, 94, 73, 24, 41, 49, 6, 45, 3, 70, 10, 80,
#                  86, 74, 106, 82, 36, 21, 96, 12, 92, 71, 68, 109, 64,
#                  88, 104, 118, 53, 111, 39, 51, 110, 90, 61, 9, 34, 95,
#                  100, 108, 97, 20, 105, 121, 33, 31, 32, 103, 54, 115,
#                  112, 30, 98, 63, 43, 44, 83, 15, 18, 19, 56, 23, 22, 107,
#                  4, 11, 84, 85]
#
# val_indices = [38, 78, 8, 93, 102, 27, 29, 75, 59, 66, 57, 42, 52, 16,
#                120, 79, 17, 76, 114, 113, 69, 47, 37, 2]
#
# test_indices = [0, 48, 77, 62, 65, 35, 13, 26, 40, 99, 81, 116, 119]

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        sampler=SubsetRandomSampler(range(num_train)))
n_valid_steps = 1
n_visual_steps = 100

# Saving architectures
with open('./runs/' + logdir + '/model_gen.txt', 'w+') as f:
    f.write(str(gen))
    f.close()

with open('./runs/' + logdir + '/model_critic.txt', 'w+') as f:
    f.write(str(critic))
    f.close()

print("Start training..")
if opts.framework == "wgangp":
    # Training process : "Sequential" Wasserstein GAN-GP/LP
    total_iterations = 0
    for epoch in range(num_epochs):
        gen.train()
        for (real, _, _, _) in dataloader:
            total_iterations += 1
            optim_critic.zero_grad()
            real = real.to(device)
            noise = torch.randn(batch_size, stick_length,
                                input_vector_size, device=device)
            fake = gen(noise, [stick_length] * batch_size)
            fake = fake.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            real = real.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            gp = gradient_penalty(critic, batch_size, real, fake,
                                  is_seq=True, lp=True, device=device)
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
            noise = torch.randn(batch_size, stick_length,
                                input_vector_size, device=device)
            fake = gen(noise, [stick_length] * batch_size)
            fake = fake.view(batch_size, stick_length,
                             output_size).permute(0, 2, 1)
            err_real = torch.mean(critic(real))
            err_fake = torch.mean(critic(fake))
            err_tv = tv_loss(fake)
            err_gen = err_real - err_fake + eta * err_tv
            loss_gen = err_gen.item()
            err_gen.backward()
            optim_gen.step()

            writer.add_scalar('loss_critic', -loss_critic, total_iterations)
            writer.add_scalar('loss_gen', loss_gen, total_iterations)
            writer.add_scalar('gp', gp, total_iterations)
            writer.add_scalar('w_dist', -w_dist, total_iterations)

            scheduler_critic.step()
            scheduler_gen.step()

        if (epoch + 1) % 500 == 0:
            print("Iteration: {} LossG : {} LossD : {}".format(
                total_iterations, loss_gen, loss_critic))

        # Generating sample animations of 120 frames i.e. 4.8s
        if (epoch + 1) % 5000 == 0:
            gen.eval()
            print("Generating sample animations...")
            samples = sampleseqG(gen, stick_length, fixed_noise)
            samples = np.reshape(samples, (stick_length * nb_samples, -1))
            if dataset.scaler is not None:
                samples = dataset.scaler.inverse_transform(samples)
                samples = np.reshape(samples, (nb_samples, stick_length, 23, 3))
                for s in range(nb_samples):
                    frame_to_vid(samples[s, :], sampledir +
                                 '/e{}s{}.avi'.format(epoch + 1, s), fps=25)

        if (epoch + 1) % 5000 == 0:
            # Saving weights
            torch.save(gen.state_dict(), modeldir + '/gpgen_{}.pt'.format(epoch + 1))
            torch.save(critic.state_dict(), modeldir + '/gpcritic_{}.pt'.format(epoch+1))

elif opts.framework == "gan":
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    real_label = torch.full((batch_size,), 1).to(device)
    fake_label = torch.full((batch_size,), 0).to(device)
    total_iterations = 0
    for epoch in range(num_epochs):
        gen.train()
        for (real, _, _, _) in dataloader:
            total_iterations += 1
            optim_critic.zero_grad()
            real = real.to(device)
            noise = torch.randn(batch_size, stick_length,
                                input_vector_size, device=device)
            fake = gen(noise, [stick_length] * batch_size)
            fake = fake.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            real = real.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()

            err_real = criterion(critic(real).squeeze(1), real_label)
            err_fake = criterion(critic(fake.detach()).squeeze(1), fake_label)
            err_critic = err_real + err_fake
            err_critic.backward(retain_graph=True)
            loss_critic = err_critic.item()
            optim_critic.step()

            # if total_iterations % n_critic_steps:
            #     continue

            # Train generator gen
            optim_gen.zero_grad()
            noise = torch.randn(batch_size, stick_length,
                                input_vector_size, device=device)
            fake = gen(noise, [stick_length] * batch_size)
            fake = fake.view(batch_size, stick_length,
                             output_size).permute(0, 2, 1)
            err_fake = criterion(critic(fake).squeeze(1), real_label)
            err_tv = tv_loss(fake)
            err_gen = err_fake + eta * err_tv
            loss_gen = err_gen.item()
            err_gen.backward()
            optim_gen.step()

            writer.add_scalar('loss_D', loss_critic, total_iterations)
            writer.add_scalar('loss_G', loss_gen, total_iterations)

            scheduler_critic.step()
            scheduler_gen.step()

        if (epoch + 1) % 500 == 0:
            print("Iteration: {} LossG : {} LossD : {}".format(
                total_iterations, loss_gen, loss_critic))

        # Generating sample animations of 120 frames i.e. 4.8s
        if (epoch + 1) % 5000 == 0:
            gen.eval()
            print("Generating sample animations...")
            samples = sampleseqG(gen, stick_length, fixed_noise)
            samples = np.reshape(samples, (stick_length * nb_samples, -1))
            if dataset.scaler is not None:
                samples = dataset.scaler.inverse_transform(samples)
                samples = np.reshape(samples, (nb_samples, stick_length, 23, 3))
                for s in range(nb_samples):
                    frame_to_vid(samples[s, :], sampledir +
                                 '/e{}s{}.avi'.format(epoch + 1, s), fps=25)
else:
    raise ValueError("Please state existing framework")

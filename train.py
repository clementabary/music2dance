from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset, SequenceDataset
from utils import collate_fn, slice_audio_batch
from losses import gradient_penalty, tv_loss
import torch
import torch.optim as optim
import os
import datetime
import numpy as np
from models.withaudio import SequenceGenerator, SequenceDiscriminator
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

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

# Training hyper-parameters
num_train = 60               # 61 in total (122 with mirror augmentation) FIXME: Waltz is corrupted
num_epochs = 100000          # approximately 8h on Colab
batch_size = 30              # FIXME: needs to divide num_train
window_size = 0.2            # width of audio window (in s)
nb_samples = 5               # number of samples generated during training
seq_length = 4.8             # fixed length of sequence for training
gamma = 10                   # gradient penalty coefficient
beta = 100                   # l1 regularization coefficient
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
n_critic_steps = 5           # times critic is updated before generator
freeze_epoch = False         # number of epochs before freezing decoder
init_kernel = 25             # kernel of first 1D convolution in critic
code_size = 100              # code size before concatening audio-stick embeddings

config = {'audio_rate': 44100, 'video_rate': 25, 'feat_size': window_size, 'seq_length': seq_length}

# Creating datasets
# FIXME: if use augmentation, change scaler
sticks = StickDataset('./data/fixed_centered_skeletons.npy',
                      resume=True,
                      normalize='minmax')

dataset = SequenceDataset('./Music-to-Dance-Motion-Synthesis-master', config,
                          scaler=sticks.scaler, withaudio=True)
dataset.resample_audio(16000)
dataset.truncate()

# Sequence parameters
cutting_stride = dataset.ratio
audio_feat_samples = int(window_size * dataset.aud_rate)
pad_samples = audio_feat_samples - cutting_stride

# Creating models
gen = SequenceGenerator(window_size, input_vector_size, latent_vector_size,
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
# BE CAREFUL: check if len train/valid can be divided by batch_size !
dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_split = .2
random_seed = 42
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collate_fn, sampler=train_sampler)
valid_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                              sampler=train_sampler)
n_valid_steps = 1
criterion = torch.nn.L1Loss(reduction='mean')

# Training process : "Sequential" Wasserstein GAN-GP/LP
total_iterations = 0
for epoch in range(num_epochs):
    gen.train()
    for (real, _, audio, _, _) in train_dataloader:
        total_iterations += 1
        real = real.to(device)
        audio = audio.to(device)
        optim_critic.zero_grad()
        audio_slices = slice_audio_batch(audio, audio_feat_samples,
                                         cutting_stride, pad_samples)
        audio = audio.unsqueeze(1)
        fake = gen(audio_slices, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        real = real.view(batch_size, seq_length, output_size).permute(
            0, 2, 1).contiguous()
        gp = gradient_penalty(critic, batch_size, real, fake, audio,
                              is_seq=True, lp=True, device=device)
        err_real = torch.mean(critic(real, audio))
        err_fake = torch.mean(critic(fake.detach(), audio))
        err_critic = err_fake - err_real + gamma * gp
        w_dist = err_fake - err_real
        loss_critic = err_critic.item()
        err_critic.backward(retain_graph=True)
        optim_critic.step()

        if total_iterations % n_critic_steps:
            continue

        # Train generator gen
        optim_gen.zero_grad()
        fake = gen(audio_slices, [seq_length] * batch_size)
        fake = fake.view(batch_size, seq_length,
                         output_size).permute(0, 2, 1)
        err_l1 = criterion(real, fake)
        err_real = torch.mean(critic(real, audio))
        err_fake = torch.mean(critic(fake, audio))
        err_tv = tv_loss(fake)
        err_gen = err_real - err_fake + beta * err_l1 + eta * err_tv
        loss_gen = err_gen.item()
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', loss_critic, total_iterations)
        writer.add_scalar('loss_gen', loss_gen, total_iterations)
        writer.add_scalar('gp', gp, total_iterations)
        writer.add_scalar('w_dist', w_dist, total_iterations)
        writer.add_scalar('l1_loss_train', err_l1, total_iterations)

        scheduler_critic.step()
        scheduler_gen.step()

    if epoch % n_valid_steps == 0:
        gen.eval()
        val_loss = []
        for (real, _, audio, _, _) in valid_dataloader:
            real = real.to(device)
            fake = gen(audio_slices, [seq_length] * batch_size)
            fake = fake.view(batch_size, seq_length, output_size).permute(0, 2, 1)
            val_loss.append(criterion(real, fake).item())
        err_val = np.mean(val_loss)
        writer.add_scalar('l1_loss_val', err_val, total_iterations)

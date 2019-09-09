from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
# from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset, SequenceDataset
from utils import collate_fn, slice_audio_batch
from losses import gradient_penalty, tv_loss
import torch
import torch.optim as optim
import os
import datetime
import numpy as np
from archis.default import SequenceGenerator, SequenceDiscriminator
from archis.default import AblatedSequenceDiscriminator
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import json

# Check for GPU access
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose config file")
parser.add_argument("-d", "--device", type=int, help="choose gpu id")
parser.add_argument("-n", "--name", type=str, help="name experiment")
opts = parser.parse_args()

device_idx = opts.device
device = torch.device("cuda:" + str(device_idx)
                      if torch.cuda.is_available() else "cpu")

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
window_size = cfg['window_size']
seq_length = cfg['seq_length']
gamma = cfg['gamma']
beta = cfg['beta']
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
noise_size = cfg['noise_size']
code_size = cfg['code_size']
enc_type = cfg['enc_type']
ablated = cfg['ablated']
init_kernel = cfg['init_kernel']
activ = cfg['activ']

# Creating datasets
print("Loading sticks and sequences datasets...")
sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=True)
dataset.truncate()

# Sequence parameters
stick_length = dataset.stick_length
cutting_stride = dataset.ratio
audio_feat_samples = int(window_size * dataset.aud_rate)
pad_samples = audio_feat_samples - cutting_stride

# Creating models
print("Loading models..")
gen = SequenceGenerator(audio_feat_samples, input_vector_size, latent_vector_size,
                        size, output_size, noise_size, nblocks_gen, n_cells,
                        enc_type, activ, device)
# print(sum(p.numel() for p in gen.parameters() if p.requires_grad))
if ablated:
    critic = AblatedSequenceDiscriminator(output_size, channels, code_size,
                                          dataset.stick_length, init_ker=init_kernel,
                                          activ=activ, device=device)
else:
    critic = SequenceDiscriminator(output_size, channels, code_size,
                                   dataset.stick_length, init_ker=init_kernel,
                                   activ=activ, device=device)
# print(sum(p.numel() for p in critic.parameters() if p.requires_grad))

# Creating optimizers and schedulers
optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)
optim_gen = optim.Adam(gen.parameters(), lr=lr_gen)

# Logging
print("Prepare for logging..")
writer = SummaryWriter('./runs/' + logdir + '/logging')
sampledir = './runs/' + logdir + '/samples'
os.makedirs(sampledir)
modeldir = './runs/' + logdir + '/models'
os.makedirs(modeldir)

# Loading data
validation_split = .2
test_split = .5
random_seed = 14
dataset_size = len(dataset)
indices = list(range(dataset_size))
vsplit = int(np.floor(validation_split * dataset_size))
tsplit = int(np.floor(test_split * vsplit))
np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[vsplit:], indices[tsplit:vsplit]
test_indices = indices[:tsplit]
samples_dict = {'train_samples': [dataset.dirs[idx] for idx in train_indices],
                'val_samples': [dataset.dirs[idx] for idx in val_indices],
                'test_samples': [dataset.dirs[idx] for idx in test_indices]}

with open('./runs/' + logdir + '/trainvaltest_samples.json', 'w+') as f:
    json.dump(samples_dict, f)

train_class_sample_count = np.unique(
    dataset.labels[train_indices], return_counts=True)[1]
train_weight = 1. / train_class_sample_count
train_samples_weight = train_weight[dataset.labels[train_indices]]
train_sampler = WeightedRandomSampler(
    train_samples_weight, len(train_samples_weight))

val_class_sample_count = np.unique(
    dataset.labels[val_indices], return_counts=True)[1]
val_weight = 1. / val_class_sample_count
val_samples_weight = val_weight[dataset.labels[val_indices]]
val_sampler = WeightedRandomSampler(
    val_samples_weight, len(val_samples_weight))

train_dict = {'sequences': [dataset.sequences[idx] for idx in train_indices],
              'labels': [dataset.labels[idx] for idx in train_indices],
              'dirs': [dataset.dirs[idx] for idx in train_indices],
              'musics': [dataset.musics[idx] for idx in train_indices]}
val_dict = {'sequences': [dataset.sequences[idx] for idx in val_indices],
            'labels': [dataset.labels[idx] for idx in val_indices],
            'dirs': [dataset.dirs[idx] for idx in val_indices],
            'musics': [dataset.musics[idx] for idx in val_indices]}
train_dataset = SequenceDataset(
    train_dict, cfg['dataset'], resume=True, withaudio=True)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                          sampler=train_sampler, collate_fn=collate_fn)
val_dataset = SequenceDataset(
    val_dict, cfg['dataset'], resume=True, withaudio=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                        sampler=val_sampler, collate_fn=collate_fn)
# train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
#                           sampler=SubsetRandomSampler(range(49)))
# val_loader = DataLoader(dataset, batch_size=6, collate_fn=collate_fn,
#                         sampler=SubsetRandomSampler(range(49, 61)))

n_valid_steps = 1
n_visual_steps = 100
criterion = torch.nn.L1Loss(reduction='mean')

# Saving architectures
with open('./runs/' + logdir + '/model_gen.txt', 'w+') as f:
    f.write(str(gen))
    f.close()

with open('./runs/' + logdir + '/model_critic.txt', 'w+') as f:
    f.write(str(critic))
    f.close()

print("Start training..")
# Training process : "Sequential" Wasserstein GAN-GP/LP with audio conditioning
total_iterations = 0
for epoch in range(num_epochs):
    gen.train()
    for (real, _, audio, _, _) in train_loader:
        total_iterations += 1
        optim_critic.zero_grad()
        audio_slices = slice_audio_batch(audio, audio_feat_samples,
                                         cutting_stride, pad_samples)
        real = real.to(device)
        audio = audio.to(device)
        audio_slices = audio_slices.to(device)
        audio = audio.unsqueeze(1)
        fake = gen(audio_slices, [stick_length] * batch_size)
        fake = fake.view(batch_size, stick_length, output_size).permute(
            0, 2, 1).contiguous()
        real = real.view(batch_size, stick_length, output_size).permute(
            0, 2, 1).contiguous()
        if ablated:
            gp = gradient_penalty(critic, batch_size, real, fake,
                                  is_seq=True, lp=False, device=device)
        else:
            gp = gradient_penalty(critic, batch_size, real, fake, audio,
                                  is_seq=True, lp=False, device=device)
        if ablated:
            err_real = torch.mean(critic(real))
            err_fake = torch.mean(critic(fake.detach()))
        else:
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
        fake = gen(audio_slices, [stick_length] * batch_size)
        fake = fake.view(batch_size, stick_length,
                         output_size).permute(0, 2, 1)
        err_l1 = criterion(real, fake)
        if ablated:
            err_real = torch.mean(critic(real))
            err_fake = torch.mean(critic(fake))
        else:
            err_real = torch.mean(critic(real, audio))
            err_fake = torch.mean(critic(fake, audio))
        err_tv = tv_loss(fake)
        err_gen = err_real - err_fake + beta * err_l1 + eta * err_tv
        loss_gen = err_gen.item()
        err_gen.backward()
        optim_gen.step()

        writer.add_scalar('loss_critic', -loss_critic, total_iterations)
        writer.add_scalar('loss_gen', loss_gen, total_iterations)
        writer.add_scalar('gp', gp, total_iterations)
        writer.add_scalar('w_dist', -w_dist, total_iterations)
        writer.add_scalar('l1_loss_train', err_l1, total_iterations)

    if epoch % n_valid_steps == 0:
        gen.eval()
        e_val_loss = []
        with torch.no_grad():
            for (real, _, audio, _, _) in val_loader:
                real = real.to(device)
                audio_slices = slice_audio_batch(audio, audio_feat_samples,
                                                 cutting_stride, pad_samples)
                audio_slices = audio_slices.to(device)
                real = real.view(
                    real.size()[0], stick_length, output_size).permute(0, 2, 1)
                fake = gen(audio_slices, [stick_length] * real.size()[0])
                fake = fake.view(
                    real.size()[0], stick_length, output_size).permute(0, 2, 1)
                e_val_loss.append(criterion(real, fake).item())
            e_val_loss = np.mean(e_val_loss)
            writer.add_scalar('l1_loss_val', e_val_loss, total_iterations)

    if (epoch + 1) % 500 == 0:
        print("Iteration: {} LossG : {} LossD : {} L1 train : {} L1 val : {}".format(
            total_iterations, loss_gen, loss_critic, err_l1, e_val_loss))

    if (epoch + 1) <= 1000 and (epoch + 1) % 100 == 0:
        torch.save(gen.state_dict(), modeldir +
                   '/gpgen_{}.pt'.format(epoch + 1))

    if (epoch + 1) % 5000 == 0:
        # Saving weights
        torch.save(gen.state_dict(), modeldir +
                   '/gpgen_{}.pt'.format(epoch + 1))
        torch.save(critic.state_dict(), modeldir +
                   '/gpcritic_{}.pt'.format(epoch + 1))

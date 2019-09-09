from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset, SequenceDataset
from utils import collate_fn
from utils import gen_rand_noise_with_label
from losses import gradient_penalty, tv_loss
import torch
import torch.optim as optim
import os
import datetime
from phase2.archis.conditional import SequenceGenerator, SequenceDiscriminator
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
device = torch.device("cuda:" + str(device_idx)
                      if torch.cuda.is_available() else "cpu")

with open(opts.config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

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
num_classes = 4

# Creating datasets
print("Loading sticks and sequences datasets...")
sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=False)

stick_length = dataset.stick_length
fixed_noise = torch.randn(nb_samples, stick_length,
                          input_vector_size, device=device)

# Creating models
print("Loading models..")
gen = SequenceGenerator(input_vector_size, latent_vector_size,
                        size, output_size, nblocks_gen, n_cells, device)
critic = SequenceDiscriminator(output_size, channels, dataset.stick_length,
                               num_classes, init_kernel, nblocks_critic, device)

# Creating optimizers
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
# BE CAREFUL: check if len train/valid can be divided by batch_size !
print("Loading data..")
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        sampler=SubsetRandomSampler(range(num_train)))
n_valid_steps = 1
aux_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

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
        for (real_sticks, _, real_label, _) in dataloader:
            total_iterations += 1
            optim_critic.zero_grad()
            real_sticks = real_sticks.to(device)
            real_label = real_label.to(device)
            noise, fake_label = gen_rand_noise_with_label(num_classes, batch_size,
                                                          input_vector_size, stick_length,
                                                          device=device)
            fake_sticks = gen(noise, [stick_length] * batch_size)
            fake_sticks = fake_sticks.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            real_sticks = real_sticks.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            gp = gradient_penalty(critic, batch_size, real_sticks, fake_sticks,
                                  is_seq=True, is_cond=True, lp=True, device=device)

            err_real, aux_real = critic(real_sticks)
            err_real = torch.mean(err_real)
            aux_err_real = aux_criterion(aux_real, real_label)

            err_fake, aux_fake = critic(fake_sticks.detach())
            err_fake = torch.mean(err_fake)
            aux_err_fake = aux_criterion(aux_fake, fake_label)
            err_ac = aux_err_real + aux_err_fake

            err_critic = err_fake - err_real + err_ac + gamma * gp
            w_dist = err_fake - err_real
            loss_critic = err_critic.item()
            err_critic.backward(retain_graph=True)
            optim_critic.step()

            if total_iterations % n_critic_steps:
                continue

            # Train generator gen
            optim_gen.zero_grad()
            noise, fake_label = gen_rand_noise_with_label(num_classes, batch_size,
                                                          input_vector_size, stick_length,
                                                          device=device)
            fake_sticks = gen(noise, [stick_length] * batch_size)
            fake_sticks = fake_sticks.view(batch_size, stick_length,
                                           output_size).permute(0, 2, 1)
            err_real, aux_real = critic(real_sticks)
            err_real = torch.mean(err_real)
            aux_err_real = aux_criterion(aux_real, real_label)

            err_fake, aux_fake = critic(fake_sticks)
            err_fake = torch.mean(err_fake)
            aux_err_fake = aux_criterion(aux_fake, fake_label)
            err_ac = aux_err_real + aux_err_fake
            err_tv = tv_loss(fake_sticks)

            err_gen = err_real - err_fake + err_ac + eta * err_tv
            loss_gen = err_gen.item()
            err_gen.backward()
            optim_gen.step()

            writer.add_scalar('loss_critic', -loss_critic, total_iterations)
            writer.add_scalar('loss_gen', loss_gen, total_iterations)
            writer.add_scalar('gp', gp, total_iterations)
            writer.add_scalar('w_dist', -w_dist, total_iterations)

        if (epoch + 1) % 500 == 0:
            print("Iteration: {} LossG : {} LossD : {}".format(
                total_iterations, loss_gen, loss_critic))

        if (epoch + 1) % 5000 == 0:
            # Saving weights
            torch.save(gen.state_dict(), modeldir +
                       '/gpgen_{}.pt'.format(epoch + 1))
            torch.save(critic.state_dict(), modeldir +
                       '/gpcritic_{}.pt'.format(epoch + 1))

elif opts.framework == "gan":
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    real_label_gan = torch.full((batch_size,), 1).to(device)
    fake_label_gan = torch.full((batch_size,), 0).to(device)
    total_iterations = 0
    for epoch in range(num_epochs):
        gen.train()
        for (real_sticks, _, real_label, _) in dataloader:
            total_iterations += 1
            optim_critic.zero_grad()
            real_sticks = real_sticks.to(device)
            real_label = real_label.to(device)
            noise, fake_label = gen_rand_noise_with_label(num_classes, batch_size,
                                                          input_vector_size, stick_length,
                                                          device=device)
            fake_sticks = gen(noise, [stick_length] * batch_size)
            fake_sticks = fake_sticks.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()
            real_sticks = real_sticks.view(batch_size, stick_length, output_size).permute(
                0, 2, 1).contiguous()

            err_real, aux_real = critic(real_sticks)
            err_real = criterion(err_real.squeeze(1), real_label_gan)
            aux_err_real = aux_criterion(aux_real, real_label)

            err_fake, aux_fake = critic(fake_sticks.detach())
            err_fake = criterion(err_fake.squeeze(1), fake_label_gan)
            aux_err_fake = aux_criterion(aux_fake, fake_label)
            err_ac = aux_err_real + aux_err_fake

            err_critic = err_fake - err_real + err_ac
            loss_critic = err_critic.item()
            err_critic.backward(retain_graph=True)
            optim_critic.step()

            # if total_iterations % n_critic_steps:
            #     continue

            # Train generator gen
            optim_gen.zero_grad()
            noise, fake_label = gen_rand_noise_with_label(num_classes, batch_size,
                                                          input_vector_size, stick_length,
                                                          device=device)
            fake_sticks = gen(noise, [stick_length] * batch_size)
            fake_sticks = fake_sticks.view(batch_size, stick_length,
                                           output_size).permute(0, 2, 1)

            err_fake, aux_fake = critic(fake_sticks)
            err_fake = criterion(err_fake.squeeze(1), real_label_gan)
            aux_err_fake = aux_criterion(aux_fake, fake_label)
            err_ac = aux_err_fake
            err_tv = tv_loss(fake_sticks)

            err_gen = err_fake + err_ac + eta * err_tv
            loss_gen = err_gen.item()
            err_gen.backward()
            optim_gen.step()

            writer.add_scalar('loss_disc', loss_critic, total_iterations)
            writer.add_scalar('loss_gen', loss_gen, total_iterations)

        if (epoch + 1) % 500 == 0:
            print("Iteration: {} LossG : {} LossD : {}".format(
                total_iterations, loss_gen, loss_critic))

        if (epoch + 1) % 5000 == 0:
            # Saving weights
            torch.save(gen.state_dict(), modeldir +
                       '/gen_{}.pt'.format(epoch + 1))
            torch.save(critic.state_dict(), modeldir +
                       '/disc_{}.pt'.format(epoch + 1))

else:
    raise ValueError("Please state existing framework")

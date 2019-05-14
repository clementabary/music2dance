from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import StickDataset
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.mlp import Generator, Discriminator


# TODO: check normalization techniques (some EDA as well ?)
# check architectures (layers)
# check gradient visualizations
# find visualization tool for stick figures
# adapt code for colab compatibility (CUDA GPU for at scale learning)


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(datasetf)

m, s = dataset.statistics()
# dataset = StickDataset(datasetf, 'minmax')

# Training hyper-parameters
num_train = 10000  # 70000
num_epochs = 10
batch_size = 8

# Loading dataset
writer = SummaryWriter('./runs/wgan')
dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(range(num_train)))
n_iter = len(dataloader)
n_critic_steps = 5
threshold = 0.1

# Creating models
latent_vector_size = 10
real_label = torch.full((batch_size,), 1)
fake_label = torch.full((batch_size,), 0)
fixed_noise = torch.randn(batch_size, latent_vector_size)

gen = Generator(latent_vector_size)
sum(p.numel() for p in gen.parameters() if p.requires_grad)

critic = Discriminator(69)
sum(p.numel() for p in critic.parameters() if p.requires_grad)

optim_critic = optim.Adam(critic.parameters(), lr=0.0002)
optim_gen = optim.Adam(gen.parameters(), lr=0.0002)

losses_critic = []
losses_gen = []

# Training process : vanilla GAN
for epoch in range(num_epochs):
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
        output = critic(fake)
        err_gen = - torch.mean(output)
        loss_gen += err_gen.item()
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

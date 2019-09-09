from utils import StickDataset, SequenceDataset, collate_fn
from models import RecurrentDanceClassifier
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose config file")
parser.add_argument("-d", "--device", type=int, help="choose gpu id")
parser.add_argument("-n", "--name", type=str, help="choose name of experiment")

opts = parser.parse_args()

if not os.path.exists('./logs/' + opts.name):
    os.makedirs('./logs/' + opts.name)

with open(opts.config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)
device_idx = opts.device
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

print("Loading datasets..")
sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=True)
dataset.truncate()

batch_size = cfg['batch_size']
num_epochs = cfg['num_epochs']
n_valid_steps = 1

# train_indices = [101, 60, 58, 55, 50, 46, 7, 5, 67, 14, 72, 25, 91, 87,
#                  89, 1, 117, 94, 73, 24, 41, 49, 6, 45, 3, 70, 10, 80,
#                  86, 74, 106, 82, 36, 21, 96, 12, 92, 71, 68, 109, 64,
#                  88, 104, 118, 53, 111, 39, 51, 110, 90, 61, 9, 34, 95,
#                  100, 108, 97, 20, 105, 121, 33, 31, 32, 103, 54, 115,
#                  112, 30, 98, 63, 43, 44, 83, 15, 18, 19, 56, 23, 22, 107,
#                  4, 11, 84, 85]
# val_indices = [38, 78, 8, 93, 102, 27, 29, 75, 59, 66, 57, 42, 52, 16,
#                120, 79, 17, 76, 114, 113, 69, 47, 37, 2]
# test_indices = [0, 48, 77, 62, 65, 35, 13, 26, 40, 99, 81, 116, 119]
#
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
#
# train_dataloader = DataLoader(dataset, batch_size=batch_size,
#                               collate_fn=collate_fn, sampler=train_sampler)
# valid_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
#                               sampler=valid_sampler)

validation_split = .2
test_split = .5
random_seed = 19
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

with open('./logs/' + opts.name + '/trainvaltest_samples.json', 'w+') as f:
    json.dump(samples_dict, f)

train_class_sample_count = np.unique(dataset.labels[train_indices], return_counts=True)[1]
train_weight = 1. / train_class_sample_count
train_samples_weight = train_weight[dataset.labels[train_indices]]
train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

val_class_sample_count = np.unique(dataset.labels[val_indices], return_counts=True)[1]
val_weight = 1. / val_class_sample_count
val_samples_weight = val_weight[dataset.labels[val_indices]]
val_sampler = WeightedRandomSampler(val_samples_weight, len(val_samples_weight))

train_dict = {'sequences': [dataset.sequences[idx] for idx in train_indices],
              'labels': [dataset.labels[idx] for idx in train_indices],
              'dirs': [dataset.dirs[idx] for idx in train_indices],
              'musics': [dataset.musics[idx] for idx in train_indices]}
val_dict = {'sequences': [dataset.sequences[idx] for idx in val_indices],
            'labels': [dataset.labels[idx] for idx in val_indices],
            'dirs': [dataset.dirs[idx] for idx in val_indices],
            'musics': [dataset.musics[idx] for idx in val_indices]}
train_dataset = SequenceDataset(train_dict, cfg['dataset'], resume=True, withaudio=True)
train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              sampler=train_sampler, collate_fn=collate_fn)
val_dataset = SequenceDataset(val_dict, cfg['dataset'], resume=True, withaudio=True)
valid_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset),
                              sampler=val_sampler, collate_fn=collate_fn)

model = RecurrentDanceClassifier(69, 128, 4).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
writer = SummaryWriter('./logs/' + opts.name)


print("Start training..")
for epoch in range(num_epochs):
    model.train()
    train_loss = []
    for (sticks, _, _, true_label, _) in train_dataloader:

        sticks = sticks.to(device)
        sticks = sticks.view(sticks.shape[0], 120, 69).permute(
            0, 2, 1).contiguous()
        true_label = true_label.to(device)

        pred_label = model(sticks)
        err_train = criterion(pred_label, true_label.view(-1))
        train_loss.append(err_train.item())

        optim.zero_grad()
        err_train.backward()
        optim.step()

    err_train = np.mean(train_loss)
    writer.add_scalar('loss_train', err_train.item(), epoch)

    if epoch % n_valid_steps == 0:
        model.eval()
        val_loss = []
        for (sticks, _, _, true_label, _) in valid_dataloader:
            sticks = sticks.to(device)
            sticks = sticks.view(sticks.shape[0], 120, 69).permute(
                0, 2, 1).contiguous()
            true_label = true_label.to(device)
            val_loss.append(criterion(model(sticks), true_label.view(-1)).item())
        err_val = np.mean(val_loss)
        writer.add_scalar('loss_val', err_val, epoch)

torch.save(model.state_dict(), './logs/' + opts.name + '/weights.pt')

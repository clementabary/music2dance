from phase3.archis.default import SequenceGenerator
from utils import slice_audio_batch, collate_fn
from visualize import frame_to_vid
from models import RecurrentDanceClassifier
from utils import StickDataset, SequenceDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import sampleseqG
from losses import jerkiness
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import yaml
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose config file")
parser.add_argument("-l", "--logdir", type=str, help="choose logging directory")
opts = parser.parse_args()

# Configuration
sample_dir = opts.logdir + '/samples'
model_weights = opts.logdir + '/models/gpgen_100000.pt'

with open(opts.config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

# Dataset
data_folder = './Music-to-Dance-Motion-Synthesis-master'
sticks = StickDataset(data_folder, normalize='minmax')
dataset = SequenceDataset(data_folder, cfg['dataset'],
                          dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=True)
dataset.truncate()

# Sequence parameters
window_size = cfg['window_size']
nblocks_gen = cfg['nblocks_gen']
input_vector_size = cfg['input_vector_size']
latent_vector_size = cfg['latent_vector_size']
n_cells = cfg['n_cells']
size = cfg['size']
output_size = cfg['output_size']
enc_type = cfg['enc-type']
activ = cfg['activ']
seq_length = int(25 * 30)
stick_length = dataset.stick_length
cutting_stride = dataset.ratio
audio_feat_samples = int(window_size * dataset.aud_rate)
pad_samples = audio_feat_samples - cutting_stride

with open(opts.logdir + '/trainvaltest_samples.json') as f:
    data = json.load(f)

val_indices = [7, 9, 21, 22, 30, 54]  # data['val_samples']
batch_size = 1
sampler = SubsetRandomSampler(val_indices)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_fn, sampler=sampler)

model = SequenceGenerator(window_size, input_vector_size, latent_vector_size,
                          size, output_size, nblocks_gen, n_cells,
                          enc_type="default", activ="", device='cpu')
model.load_state_dict(torch.load(model_weights, map_location='cpu'))

# Visualization step
seq_length = int(25 * 30)
samples = sampleseqG(model, seq_length)
samples = np.reshape(samples, (seq_length, 69))
fake_sticks = dataset.scaler.inverse_transform(samples)
fake_sticks = np.reshape(fake_sticks, (seq_length, 23, 3))
frame_to_vid(fake_sticks, sample_dir + '/sample.avi', fps=25)

# Jerkiness evaluation
global_i = 0
real_jerk = []
fake_jerk = []
for _ in range(20):
    for i, (real_sticks, _, audio, label, _) in enumerate(dataloader):
        global_i += 1
        real_sticks = real_sticks.view(-1, stick_length, 69).numpy()
        audio_slice = slice_audio_batch(audio, audio_feat_samples, cutting_stride,
                                        pad_samples, 'cpu')
        with torch.no_grad():
            fake_sticks = model(
                audio_slice, [stick_length] * batch_size).numpy()

        fake_sticks = dataset.scaler.inverse_transform(fake_sticks)
        fake_jerk.append(jerkiness(torch.from_numpy(
            fake_sticks).view(1, -1, 69).permute(0, 2, 1)))
        # fake_sticks = np.reshape(fake_sticks, (stick_length, 23, 3))

        real_sticks = np.reshape(real_sticks, (stick_length, 69))
        real_sticks = dataset.scaler.inverse_transform(real_sticks)
        real_jerk.append(jerkiness(torch.from_numpy(
            real_sticks).view(1, -1, 69).permute(0, 2, 1)))

real_m_jerk = torch.stack(real_jerk).mean()
real_s_jerk = torch.stack(real_jerk).std()
fake_m_jerk = torch.stack(fake_jerk).mean()
fake_s_jerk = torch.stack(fake_jerk).std()


# Style classification : audio-related realism
classifier = RecurrentDanceClassifier(69, 128, 4)
classifier.load_state_dict(torch.load(
    '../dance_classification/logs/type2/weights.pt', map_location='cpu'))
classifier.eval()

softmax = torch.nn.Softmax(1)
real_labels = []
classif_labels = []
fake_labels = []
for _ in range(10):
    with torch.no_grad():
        for (real_sticks, _, audio, real_label, _) in dataloader:
            real_sticks = real_sticks.view(real_sticks.shape[0], 120, 69).permute(
                0, 2, 1).contiguous()
            audio_slice = slice_audio_batch(audio, audio_feat_samples, cutting_stride,
                                            pad_samples, 'cpu')
            real_labels.append(real_label)
            classif_label = softmax(classifier(real_sticks))
            classif_labels.append(torch.argmax(classif_label, 1))
            with torch.no_grad():
                fake_sticks = model(audio_slice, [stick_length] * batch_size)
            fake_sticks = fake_sticks.view(-1, 120, 69).permute(
                0, 2, 1).contiguous()
            fake_label = softmax(classifier(fake_sticks))
            fake_labels.append(torch.argmax(fake_label, 1))

cm = confusion_matrix(classif_labels, fake_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
xticklabels = yticklabels = ['chacha', 'rumba', 'tango', 'waltz']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, cmap="YlGnBu", ax=ax, annot=True,
            xticklabels=xticklabels, yticklabels=yticklabels)
plt.tight_layout()
plt.savefig(opts.logdir + '/cmtrain.png')

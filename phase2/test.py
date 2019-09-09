from phase2.models.tempconv import SequenceGenerator
from utils import StickDataset
from dance_classification.archis import RecurrentDanceClassifier
from utils import gen_rand_noise_with_label  # , one_hot_encode
from sklearn.metrics import confusion_matrix
from utils import sampleseqG
# from losses import jerkiness
import torch
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose configuration")
parser.add_argument("-l", "--logdir", type=str, help="choose logging directory")
parser.addd
opts = parser.parse_args()

dataset = StickDataset('./Music-to-Dance-Motion-Synthesis-master', normalize='minmax')

with open(opts.config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)


seq_length = cfg['seq_length']
nblocks_gen = cfg['nblocks_gen']
input_vector_size = cfg['input_vector_size']
latent_vector_size = cfg['latent_vector_size']
n_cells = cfg['n_cells']
size = cfg['size']
channels = cfg['channels']
output_size = cfg['output_size']

nb_samples = 5

dir = opts.logdir
modeldir = dir + '/models'
sampledir = dir + '/samples'
model = SequenceGenerator(input_vector_size, latent_vector_size,
                          size, output_size, nblocks_gen, n_cells)
model.load_state_dict(torch.load(modeldir + '/gpgen_100000.pt', map_location='cpu'))
model.eval()
seq_length = int(25*4.8)

classifier = RecurrentDanceClassifier(69, 128, 4)
weights_file = './dance_classification/logs/default/weights.pt'
classifier.load_state_dict(torch.load(weights_file, map_location='cpu'))

softmax = torch.nn.Softmax(1)
model.eval()

classif_labels = []
real_labels = []
for _ in range(1000):
    noise, label = gen_rand_noise_with_label(4, 1, seq_length, input_vector_size)
    real_labels.append(label)
    samples = sampleseqG(model, seq_length, noise=noise.permute(0, 2, 1))
    samples = torch.from_numpy(samples)
    samples = samples.view(1, 120, 69).permute(0, 2, 1)

    classif_label = softmax(classifier(samples))
    classif_labels.append(torch.argmax(classif_label, 1))


cm = confusion_matrix(real_labels, classif_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


xticklabels = yticklabels = ['chacha', 'rumba', 'tango', 'waltz']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, cmap="YlGnBu", ax=ax, annot=True,
            xticklabels=xticklabels, yticklabels=yticklabels)
plt.savefig('./test_cm.png')


# for syle-conditioned generation
# label = 'W'
# ohl = one_hot_encode(label)
# fixed_noise = np.random.normal(0, 1, (1, seq_length, input_vector_size))
# noise, _ = gen_rand_noise_with_label(num_classes, 1, input_vector_size, seq_length,
#                                      label=ohl, noise=fixed_noise)
#
# model.eval()
# with torch.no_grad():
#     samples = model(noise, [seq_length]).numpy()
#
# if dataset.scaler is not None:
#     samples = dataset.scaler.inverse_transform(samples)
#     samples = np.reshape(samples, (1, seq_length, 23, 3))
#     frame_to_vid(samples[0, :], './{}.avi'.format(label), fps=25)

# for jerkiness
# jerk = []
# num_classes = 4
# label = 'W'
#
# for i in range(100):
#     ohl = one_hot_encode(label)
#     noise, _ = gen_rand_noise_with_label(num_classes, 1, input_vector_size,
#                                          seq_length, label=ohl)
#     samples = sampleseqG(model, seq_length, noise=noise.permute(0, 2, 1))
#     samples = np.reshape(samples, (seq_length, -1))
#     if dataset.scaler is not None:
#         samples = dataset.scaler.inverse_transform(samples)
#         samples = torch.from_numpy(samples).view(1, -1, 69).permute(0, 2, 1)
#         jerk.append(jerkiness(samples))
#
# m_jerk = torch.stack(jerk).mean()
# s_jerk = torch.stack(jerk).std()

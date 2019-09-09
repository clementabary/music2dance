from dance_classification.utils import StickDataset, SequenceDataset, collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dance_classification.models import RecurrentDanceClassifier
from models.tempconv import SequenceGenerator
from utils import gen_rand_noise_with_label
import torch
from utils import sampleseqG
import numpy as np
import yaml
import json
from sklearn.metrics import confusion_matrix

with open('./dance_classification/config.yaml', 'r') as yamlfile:
    cfg = yaml.load(yamlfile)
device_idx = 0  # opts.device
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=True)
dataset.truncate()
with open('./dance_classification/logs/fctype2/trainvaltest_samples.json', 'r') as f:
    data = json.load(f)

data['test_samples']
test_indices = [71, 24, 98, 95, 88, 13]
# test_indices = [1, 2, 3, 4, 5]
test_sampler = SubsetRandomSampler(test_indices)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn,
                             sampler=SubsetRandomSampler(test_indices))

model = RecurrentDanceClassifier(69, 128, 4)
model.load_state_dict(torch.load('./dance_classification/logs/type2/weights.pt', map_location='cpu'))

softmax = torch.nn.Softmax(1)

true_labels = []
pred_labels = []
for i in range(20):
    with torch.no_grad():
        model.eval()
        train_loss = []
        for (sticks, _, _, true_label, _) in test_dataloader:
            sticks = sticks.view(sticks.shape[0], 120, 69).permute(
                0, 2, 1).contiguous()
            true_labels.append(true_label)
            pred_label = softmax(model(sticks))
            pred_labels.append(torch.argmax(pred_label, 1))
            # print("True: {} | Predicted: {}".format(true_label, torch.argmax(pred_label, 1)))
        # confusion_matrix(true_label.numpy(), torch.argmax(pred_label, 1).numpy())

prl = np.asarray(torch.stack(pred_labels).flatten())
trl = np.asarray(torch.stack(true_labels).flatten())
trl
cm2= confusion_matrix(trl, prl)
cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
import seaborn as sns
import matplotlib.pyplot as plt

xticklabels = yticklabels = ['chacha', 'rumba', 'tango', 'waltz']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm2, cmap="YlGnBu", ax=ax, annot=True, xticklabels=xticklabels, yticklabels=yticklabels)
plt.savefig('./confmatrixclassif.png')

with open('./configs/config_minimal_withoutaudio.yaml', 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

from dance_classification.utils2 import StickDataset, SequenceDataset
sticks = StickDataset(cfg['folder'], normalize='minmax')
dataset = SequenceDataset(cfg['folder'], cfg['dataset'], dance_types=cfg['dance_types'],
                          scaler=sticks.scaler, withaudio=True)

seq_length = cfg['seq_length']
nblocks_gen = cfg['nblocks_gen']
input_vector_size = cfg['input_vector_size']
latent_vector_size =  cfg['latent_vector_size']
n_cells = cfg['n_cells']
size = cfg['size']
channels = cfg['channels']
output_size = cfg['output_size']

nb_samples = 5

dir = './phase2/label/20190811-133906_minimallabelseqwgan'
modeldir = dir + '/models'
sampledir = dir + '/samples'
gen = SequenceGenerator(input_vector_size, latent_vector_size,
                        size, output_size, nblocks_gen, n_cells)
gen.load_state_dict(torch.load(modeldir + '/gpgen_100000.pt', map_location='cpu'))
seq_length = int(25*4.8)

gen.eval()

true_label = 2
for _ in range(5):
    noise, label = gen_rand_noise_with_label(4, 1, seq_length, input_vector_size, label=np.array([true_label]))

    samples = sampleseqG(gen, seq_length, noise=noise.permute(0, 2, 1))

    if dataset.scaler is not None:
        samples = torch.from_numpy(samples).view(1, samples.shape[0], 69).permute(0, 2, 1).contiguous()
        pred_label = softmax(model(samples))
        print("True: {} |Predicted: {}".format(true_label, torch.argmax(pred_label, 1)))
        samples = samples.squeeze(0).permute(1, 0)
        samples = dataset.scaler.inverse_transform(samples.numpy())
        samples = np.reshape(samples, (samples.shape[0], 23, 3))
        frame_to_vid(samples, './phase2/label/test-seq{}.avi'.format(_), fps=25)

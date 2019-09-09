from utils import StickDataset, sampleG
from phase1.archis.residual import Generator
from music2dance.visualize import to_2d_graph_data, visualize_2d_graph
# from visualize import to_3d_graph_data, visualize_3d_graph
import numpy as np
import torch
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="choose configuration")
parser.add_argument("-l", "--logdir", type=str, help="choose logging directory")
parser.addd
opts = parser.parse_args()

config = opts['config']
logdir = opts['logdir']
sampledir = './phase1/logs/' + logdir + '/samples'
modeldir = './phase1/logs/' + logdir + '/models'

with open(config, 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

data_folder = '../Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(data_folder, normalize='minmax')

nb_samples = 5
nblocks_gen = cfg['nblocks_gen']
nblocks_critic = cfg['nblocks_critic']
latent_vector_size = cfg['latent_vector_size']
size = cfg['size']
output_size = cfg['output_size']

fixed_noise = torch.randn(nb_samples, latent_vector_size)
gen = Generator(latent_vector_size, size, output_size, nblocks_gen)

range_epochs = [5, 10, 15, 20, 25, 30, 35, 40]

for e in range_epochs:
    weights_file = modeldir + '/gen_{}.pt'.format(e)
    gen.load_state_dict(torch.load(weights_file, map_location='cpu'))
    gen.eval()

    print("Generating samples after {} training epochs...".format(e))
    samples = sampleG(gen, fixed_noise)
    if dataset.scaler is not None:
        samples = dataset.scaler.inverse_transform(samples)
    for s in range(nb_samples):
        trace_2d = to_2d_graph_data(np.reshape(samples[s, :], (23, 3)))
        # trace_3d = to_3d_graph_data(np.reshape(samples[s, :], (23, 3)))
        filepath = sampledir + '/e{}s{}.png'.format(e, s+1)
        visualize_2d_graph(trace_2d, save=filepath)
        # visualize_3d_graph(trace_3d, save=filepath)

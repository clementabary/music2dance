from torch.utils.data import DataLoader, TensorDataset
from utils import StickDataset, load_dataset, stickwise
import torch
import torch.nn as nn


# Load dataset
datasetf = 'Music-to-Dance-Motion-Synthesis-master'
dataset = StickDataset(datasetf)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

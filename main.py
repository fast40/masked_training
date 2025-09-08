"""
The goal of this program is to evaluate the efficacy of adding individual neurons to the loss function.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from celeba import ImageDataset

torch.manual_seed(1337)


BATCH_SIZE = 2
INPUT_DIMS = 6
HIDDEN_DIMS = 5


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(INPUT_DIMS, HIDDEN_DIMS)
        self.layer2 = nn.Linear(HIDDEN_DIMS, INPUT_DIMS)

    def forward(self, x, calculate_loss=False):
        layer1_output = self.layer1(x)

        if calculate_loss:
            masked_inputs = layer1_output.repeat(1, HIDDEN_DIMS).view(-1, HIDDEN_DIMS, HIDDEN_DIMS).tril().view(-1, HIDDEN_DIMS)
            layer2_output = self.layer2(masked_inputs)
            loss = F.mse_loss(layer2_output, x.repeat(1, HIDDEN_DIMS).view(-1, INPUT_DIMS))

            return layer2_output, loss
        else:
            layer2_output = self.layer2(layer1_output)

            return layer2_output


m = Model()

dataset = ImageDataset('celeba')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

while True:
    for batch in dataloader:
        output, loss =  m(batch, batch)

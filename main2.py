"""
Ok so the first goal here is to create a system for imshowing via matplotlib and for checkpointing.
Ok so we need to log every so often.
"""

from infra import *

import torch
from torch import optim
from torch.utils.data import DataLoader
from celeba import ImageDataset
from celebanormal import ImageDatasetNormal
from model import Model

torch.manual_seed(1337)

BATCH_SIZE = 2**6

m = Model()
m.to(DEVICE)
optimizer = optim.Adam(m.parameters(), lr=1e-3)

dataset = ImageDataset('celeba', batch_size=BATCH_SIZE)

datasetn = ImageDatasetNormal('celeba')
dataloader = DataLoader(datasetn, batch_size=BATCH_SIZE)

for step, batch in enumerate(dataset):

    batch = batch.to(DEVICE)
    optimizer.zero_grad()
    output, loss =  m(batch, calculate_loss=True)
    loss.backward()
    optimizer.step()

    if step % 1 == 0:
        logger.info(f'Loss at step {step}: {loss.item()}')
        add_metric('loss', loss.item())  # TODO: maybe also want to be able to pass x value like loss here?

    step += 1

    if step == 1500:
        m.mask = False
        m.reset_decoder()
        optimizer = optim.Adam(m.decoder.parameters())


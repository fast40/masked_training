"""
Ok so the first goal here is to create a system for imshowing via matplotlib and for checkpointing.
Ok so we need to log every so often.
"""

from infra import *
import torch
import time
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

# t = time.perf_counter()
# for step, batch in enumerate(dataloader):
#     if step % 100 == 0:
#         logger.info(f'Time for 100 batches: {time.perf_counter() - t}s')
#         logger.info(f'Step {step} ({step * BATCH_SIZE} images loaded)')
#         logger.info(f'Batch shape: {batch.shape}')
#
#         t = time.perf_counter()

t = time.perf_counter()
for step, batch in enumerate(dataset):
    if step % 100 == 0:
        logger.info(f'Time for 100 batches: {time.perf_counter() - t}s')
        logger.info(f'Step {step} ({step * BATCH_SIZE} images loaded)')
        logger.info(f'Batch shape: {batch.shape}')

        t = time.perf_counter()

    # batch = batch.to(DEVICE)
    # optimizer.zero_grad()
    # output, loss =  m(batch, calculate_loss=True)
    # loss.backward()
    # optimizer.step()
    #
    # # writer.add_scalar('training_loss', loss.item(), i)
    # # image = torch.stack([dataset[i] for i in range(16)])
    # # writer.add_images('image_input', image, i)
    # # writer.add_images('image_ouptut', m(image.to(infra.DEVICE)).detach().cpu(), i)
    # logger.info(loss.item())
    # step += 1
    #
    # if step == 1500:
    #     m.mask = False
    #     m.reset_decoder()
    #     optimizer = optim.Adam(m.decoder.parameters())


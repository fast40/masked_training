"""
The goal of this program is to evaluate the efficacy of adding individual neurons to the loss function.
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from celeba import ImageDataset

torch.manual_seed(1337)


BATCH_SIZE = 2**6
INPUT_DIMS = 3 * 128 * 128
HIDDEN_DIMS = 2**8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {DEVICE}')

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIMS, HIDDEN_DIMS * 4),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS * 4, HIDDEN_DIMS),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS * 4),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS * 4, INPUT_DIMS),
            nn.Sigmoid()
        )

        #
        # self.layer2 = nn.Linear(HIDDEN_DIMS, INPUT_DIMS)

    def forward(self, x, calculate_loss=False):
        shape = x.shape
        x = x.flatten(start_dim=1)
        encoder_output = self.encoder(x)

        if calculate_loss:
            masked_inputs = encoder_output.repeat(1, HIDDEN_DIMS).view(-1, HIDDEN_DIMS, HIDDEN_DIMS).tril().view(-1, HIDDEN_DIMS)
            # decoder_output = self.decoder(encoder_output)
            decoder_output = self.decoder(masked_inputs)
            # loss = F.mse_loss(decoder_output, x)
            loss = F.mse_loss(decoder_output, x.repeat(1, HIDDEN_DIMS).view(-1, INPUT_DIMS))

            return decoder_output.view(-1, *shape[1:]), loss
        else:
            decoder_output = self.decoder(encoder_output)

            return decoder_output.view(-1, *shape[1:])


m = Model()
m.to(DEVICE)
optimizer = optim.Adam(m.parameters(), lr=1e-3)

writer = SummaryWriter('runs/exp4')

dataset = ImageDataset('celeba')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

i = 0
while True:
    for batch in dataloader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        output, loss =  m(batch, calculate_loss=True)
        loss.backward()
        optimizer.step()

        writer.add_scalar('training_loss', loss.item(), i)
        image = torch.stack([dataset[i] for i in range(5)])
        writer.add_images('image_input', image, i)
        writer.add_images('image_ouptut', m(image.to(DEVICE)).detach().cpu(), i)
        print(loss.item())
        i += 1

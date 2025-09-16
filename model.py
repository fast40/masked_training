from torch import nn
from torch.nn import functional as F

INPUT_DIMS = 3 * 128 * 128
HIDDEN_DIMS = 2**8


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

        self.mask = True

        #
        # self.layer2 = nn.Linear(HIDDEN_DIMS, INPUT_DIMS)

    def forward(self, x, calculate_loss=False):
        shape = x.shape
        x = x.flatten(start_dim=1)
        encoder_output = self.encoder(x)

        if calculate_loss:
            if self.mask:
                masked_inputs = encoder_output.repeat(1, HIDDEN_DIMS).view(-1, HIDDEN_DIMS, HIDDEN_DIMS).tril().view(-1, HIDDEN_DIMS)

                decoder_output = self.decoder(masked_inputs)
                loss = F.mse_loss(decoder_output, x.repeat(1, HIDDEN_DIMS).view(-1, INPUT_DIMS))
            else:
                decoder_output = self.decoder(encoder_output)
                loss = F.mse_loss(decoder_output, x)

            return decoder_output.view(-1, *shape[1:]), loss
        else:
            decoder_output = self.decoder(encoder_output)

            return decoder_output.view(-1, *shape[1:])

    def reset_decoder(self):
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

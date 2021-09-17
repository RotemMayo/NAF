import torch.nn as nn
import torch.nn.functional as F


class BasicAutoEncoder(nn.Module):
    def __init__(self, encoder_layer_sizes, decoder_layer_sizes, dropout=0.001):
        """
        @param input_dim: the amount of data points to take per event
        @param latent_dim: the dimension of the latent space, this is the effective dimension of the input after
                           training
        """
        self.dropout = dropout
        self.input_dim = encoder_layer_sizes[0]
        self.latent_dim = encoder_layer_sizes[-1]
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.layer_sizes = encoder_layer_sizes + decoder_layer_sizes
        super(BasicAutoEncoder, self).__init__()
        self.encoder = []
        for i in range(len(encoder_layer_sizes) - 1):
            self.encoder.append(nn.Linear(encoder_layer_sizes[i], encoder_layer_sizes[i + 1]))
        self.encoder = nn.ModuleList(self.encoder)

        self.decoder = []
        self.decoder.append(nn.Linear(self.latent_dim, decoder_layer_sizes[0]))
        for i in range(len(decoder_layer_sizes) - 1):
            self.decoder.append(nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i + 1]))
        self.decoder = nn.ModuleList(self.decoder)

    def encode(self, x):
        for i in range(len(self.encoder)):
            x = F.dropout(F.relu(self.encoder[i](x)), p=self.dropout)
        return x

    def decode(self, x):
        for i in range(len(self.decoder)):
            x = F.dropout(F.relu(self.decoder[i](x)), p=self.dropout)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        x = self.decode(encoded)
        return x, encoded

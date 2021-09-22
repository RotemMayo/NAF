from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gc as gc
from tqdm import tqdm


def loss(output):
        """
        KLD of the output with a centered spherical gaussian distribution of the latent space dimension
        output: (mean, logvar) should be the latent space output
        """
        mu, logvar = output
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = 1 - logvar.exp() + logvar - mu**2
        #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        #KLD = torch.sum(KLD_element).mul_(-0.5)
        KLD = - 0.5 * torch.sum(KLD_element, 1)
        return KLD


class VAE(nn.Module):

    def __init__(self, layer_dims, dropout=0):
        super(VAE, self).__init__()
        self.layer_dims = layer_dims
        self.dropout = dropout
        self.encoder = nn.Sequential(nn.Linear(layer_dims[0], layer_dims[1]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(layer_dims[1], layer_dims[2]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))
        self.latent_dim = layer_dims[-1]
        self.mu = nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]),
                                nn.Tanh())
        self.logvar = nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]),
                                    nn.Tanh())
        self.criterion = loss

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    def density(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return torch.sum(- z**2, 1)

    def get_losses(self, x):
        output = self.forward(x)
        return self.criterion(output)
    
    def train_it(self, dataloader, optimizer, epochs=10):
        losses = []
        for epoch in tqdm(range(epochs)):
            for x in dataloader:
                # Forward
                batch_loss = torch.mean(self.get_losses(x))
                # Back-prop
                if not batch_loss.isinf() and not batch_loss.isnan():
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #clip gradients
                    optimizer.step()
                    losses.append(batch_loss.item())
                    gc.collect()
        return losses


def check_import():
    print("*BORAT VOICE* great success")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import gc

#DATA_SET_PATH = "C:\\Users\\rotem\\PycharmProjects\\ML4Jets\\ML4Jets-HUJI\\Data\\events_anomalydetection.h5"
DATA_SET_PATH = "/usr/people/snirgaz/rotemov/rotemov/Projects/ML4Jets-HUJI/Data/events_anomalydetection.h5"
MINI_EPOCH_SIZE = 10 ** 5  # MINI_EPOCH_SIZE / EPOCH_SIZE should be a natural number
EPOCH_SIZE = int(1.1 * 10 ** 6)  # needs to be updated to reflect exact size
BATCH_SIZE = int(1.25 * 10 ** 2)

CRITERION = nn.MSELoss()  # the loss function
SHUFFLE = False
LEARNING_RATE = 0.01
EPOCHS = 5

OUTPUT_FOLDER = "ae_models"
NAME_TEMPLATE = "in{}_lat{}_lr{}_do{}"
CHECKPOINT_TEMPLATE = "basic_ae_checkpoint_{}.pt"  # input size, latent size, learning rate, dropout
LAST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "last_{}".format(CHECKPOINT_TEMPLATE))
BEST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "best_{}".format(CHECKPOINT_TEMPLATE))
PNG_DPI = 200
TRAINING_LOSS_PNG = os.path.join(OUTPUT_FOLDER, "training_loss_{}.png")
TEST_LOSS_PNG = os.path.join(OUTPUT_FOLDER, "test_loss_{}.png")
# TODO: Add best checkpoints
# TODO: Make running script


class BasicAutoEncoder(nn.Module):

    def __init__(self, input_dim=2**8, latent_dim=2**2, dropout=0.1):
        self.dropout = dropout
        self.input_dim = input_dim
        super(BasicAutoEncoder, self).__init__()
        self.encoder = []
        dim = input_dim
        while dim / 2 >= latent_dim:
            self.encoder.append(nn.Linear(dim, int(dim / 2)))
            dim = int(dim / 2)
        self.encoder = nn.ModuleList(self.encoder)

        self.decoder = []
        while dim * 2 <= input_dim:
            self.decoder.append(nn.Linear(dim, int(dim * 2)))
            dim = int(dim * 2)
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
        x = self.encode(x)
        x = self.decode(x)
        return x


class DataLoaderGenerator:
    """
    A class for creating data loaders for the network. This allows us to control the amount of memory used by the NN
    at each time, as the dataset is fairly large.
    """
    def __init__(self, filename, chunksize, total_size, input_dim, batch_size, shuffle):
        self.filename = filename
        self.chunk_size = chunksize
        self.total_size = total_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __next__(self):
        if (self.idx + 1) * self.chunk_size > self.total_size:
            raise StopIteration
        else:
            x = pd.read_hdf(self.filename, start=self.idx * self.chunk_size, stop=(self.idx + 1) * self.chunk_size)
            x = x.rename_axis('ID').values[:, :self.input_dim]
            x = data.DataLoader(x, batch_size=self.batch_size, shuffle=self.shuffle)
            self.idx += 1
            return x

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.total_size / self.chunk_size))


def train(net, optimizer, data_loader_gen, criterion, epochs, last_cp_path, best_cp_path):
    net.eval()
    losses = []
    for epoch_num in tqdm(range(epochs)):
        data_loader_gen.reset()
        epoch_losses = []
        for data_loader in tqdm(data_loader_gen):
            mini_epoch_loss = 0
            for x in tqdm(data_loader):
                optimizer.zero_grad()  # zero the gradient buffers
                output = net(x.float())
                loss = criterion(output, x.float())
                loss.backward()
                optimizer.step()  # Does the update
                mini_epoch_loss += loss.detach().item()
                gc.collect()
            epoch_losses.append(mini_epoch_loss)
            print(mini_epoch_loss)
        losses += epoch_losses
        save(net, optimizer, sum(epoch_losses), epoch_num, last_cp_path)
    return losses


def save(net, optimizer, loss, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        },
        path)


def load(path, lr):
    checkpoint = torch.load(path)
    net = BasicAutoEncoder()
    optimizer = optim.AdamW(net.parameters(), lr)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optimizer, epoch, loss


def test(net, data_loader_gen, criterion):
    # TODO: needs to see how well this finds signals.
    net.eval()
    losses = []
    data_loader_gen.reset()
    for data_loader in tqdm(data_loader_gen):
        mini_epoch_loss = 0
        for x in tqdm(data_loader):
            output = net(x.float())
            loss = criterion(output, x.float())
            loss.backward()
            mini_epoch_loss += loss.detach().item()
            gc.collect()
        losses.append(mini_epoch_loss)
    return losses


def plot_losses(losses, path):
    plt.figure()
    plt.plot(losses, "r+")
    plt.savefig(path, dpi=PNG_DPI)


def run_net(input_dim, latent_dim, learning_rate, dropout):
    name = NAME_TEMPLATE.format(input_dim, latent_dim, learning_rate, dropout)
    last_cp_name = LAST_CHECKPOINT_PATH_TEMPLATE.format(name)
    best_cp_name = BEST_CHECKPOINT_PATH_TEMPLATE.format(name)
    if os.path.exists(last_cp_name):
        net, optimizer, epoch, _ = load(last_cp_name, learning_rate)
    else:
        net = BasicAutoEncoder(input_dim, latent_dim, dropout)
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
        epoch = 0
    data_gen = DataLoaderGenerator(DATA_SET_PATH, MINI_EPOCH_SIZE, EPOCH_SIZE, input_dim, BATCH_SIZE, SHUFFLE)

    training_losses = train(net, optimizer, data_gen, CRITERION, EPOCHS - epoch, last_cp_name, best_cp_name)
    plot_losses(training_losses, TRAINING_LOSS_PNG.format(name))

    test_losses = test(net, data_gen, CRITERION)
    plot_losses(test_losses, TEST_LOSS_PNG.format(name))


def parameter_search():
    n = np.random.random((10, 3))
    latent_dim = 2**2
    for params in n:
        learning_rate = round(10 ** (-5*params[0]), ndigits=5)
        dropout = round(10 ** (-3*params[1]), ndigits=3)
        input_dim = 2 ** (int(7*params[2])+3)
        print("lr={}, do={}, input={}".format(learning_rate, dropout, input_dim))
        run_net(input_dim, latent_dim, learning_rate, dropout)


def main():
    parameter_search()


if __name__ == "__main__":
    main()

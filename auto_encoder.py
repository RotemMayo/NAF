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


DATA_SET_PATH = "C:\\Users\\rotem\\PycharmProjects\\ML4Jets\\ML4Jets-HUJI\\Data\\events_anomalydetection.h5"
CHUNK_SIZE = 10 ** 4  # TOTAL_SIZE / CHUNK_SIZE should be a natural number
TOTAL_SIZE = int(1.1 * 10 ** 6)
BATCH_SIZE = 5 * 10 ** 2

CRITERION = nn.MSELoss()  # the loss function
SHUFFLE = False
LEARNING_RATE = 0.01
EPOCHS = 20

LAST_CHECKPOINT_PATH = "ae_models\\last_basic_ae_checkpoint.pt"
BEST_CHECKPOINT_PATH = "ae_models\\best_basic_ae_checkpoint.pt"
PNG_DPI = 100
TRAINING_LOSS_PNG = "ae_models\\training_loss.png"
TEST_LOSS_PNG = "ae_models\\test_loss.png"
# TODO: Add best checkpoints
# TODO: Make running script


class BasicAutoEncoder(nn.Module):
    INPUT_DIM = 2 ** 11
    LATENT_SPACE_DIM = 2 ** 4  # Has to be a power of 2
    DROPOUT = 0.1

    def __init__(self):
        super(BasicAutoEncoder, self).__init__()
        self.encoder = []
        dim = BasicAutoEncoder.INPUT_DIM
        while dim / 2 >= BasicAutoEncoder.LATENT_SPACE_DIM:
            self.encoder.append(nn.Linear(dim, int(dim / 2)))
            dim = int(dim / 2)
        self.encoder = nn.ModuleList(self.encoder)

        self.decoder = []
        while dim * 2 <= BasicAutoEncoder.INPUT_DIM:
            self.decoder.append(nn.Linear(dim, int(dim * 2)))
            dim = int(dim * 2)
        self.decoder = nn.ModuleList(self.decoder)

    def encode(self, x):
        for i in range(len(self.encoder)):
            x = F.dropout(F.relu(self.encoder[i](x)), p=BasicAutoEncoder.DROPOUT)
        return x

    def decode(self, x):
        for i in range(len(self.decoder)):
            x = F.dropout(F.relu(self.decoder[i](x)), p=BasicAutoEncoder.DROPOUT)
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
        loss_sum = 0
        data_loader_gen.reset()
        for data_loader in tqdm(data_loader_gen):
            for x in data_loader:
                optimizer.zero_grad()  # zero the gradient buffers
                output = net(x.float())
                loss = criterion(output, x.float())
                loss.backward()
                optimizer.step()  # Does the update
                losses.append(loss)
                loss_sum += loss
        print(loss_sum)
        save(net, optimizer, loss_sum, epoch_num, last_cp_path)
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
        for x in data_loader:
            output = net(x.float())
            loss = criterion(output, x.float())
            loss.backward()
            losses.append(loss)
    return losses


def plot_losses(losses, path):
    plt.figure()
    plt.plot(losses)
    plt.savefig(path, dpi=PNG_DPI)


def main():
    if os.path.exists(LAST_CHECKPOINT_PATH):
        net, optimizer, epoch, _ = load(LAST_CHECKPOINT_PATH, LEARNING_RATE)
    else:
        net = BasicAutoEncoder()
        optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
        epoch = 0
    data_gen = DataLoaderGenerator(DATA_SET_PATH, CHUNK_SIZE, TOTAL_SIZE, BasicAutoEncoder.INPUT_DIM,
                                   BATCH_SIZE, SHUFFLE)

    training_losses = train(net, optimizer, data_gen, CRITERION, EPOCHS - epoch, LAST_CHECKPOINT_PATH,
                            BEST_CHECKPOINT_PATH)
    plot_losses(training_losses, TRAINING_LOSS_PNG)

    test_losses = test(net, data_gen, CRITERION)
    plot_losses(test_losses, TEST_LOSS_PNG)


if __name__ == "__main__":
    main()

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
from test_model import trim_outliers, TRIM_PERCENT, NBINS
"""
Parameter search results:
input layer = 128 or 64
latent layer = 4
learning rate = 0.002
dropout = 0.001
Running for 400 epochs to see if we can get good results
"""

# DATA_SET_PATH = "C:\\Users\\rotem\\PycharmProjects\\ML4Jets\\ML4Jets-HUJI\\Data\\events_anomalydetection.h5"
DATA_SET_PATH = "/usr/people/snirgaz/rotemov/rotemov/Projects/ML4Jets-HUJI/Data/events_anomalydetection.h5"
MINI_EPOCH_SIZE = 10 ** 5  # MINI_EPOCH_SIZE / EPOCH_SIZE should be a natural number
EPOCH_SIZE = int(1.1 * 10 ** 6)  # needs to be updated to reflect exact size
BATCH_SIZE = int(1.25 * 10 ** 2)

CRITERION = nn.MSELoss()  # the loss function
SHUFFLE = False
LEARNING_RATE = 0.002
DROPOUT = 0.001
INPUT_DIM = 128
LATENT_DIM = 4
EPOCHS = 400
LOSS_IMPROVEMENT_THRESHOLD = 0.99
PATIENCE = 30

OUTPUT_FOLDER = "ae_models"
NAME_TEMPLATE = "in{}_lat{}_lr{}_do{}"
CHECKPOINT_TEMPLATE = "basic_ae_checkpoint_{}.pt"  # input size, latent size, learning rate, dropout
LAST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "last_{}".format(CHECKPOINT_TEMPLATE))
BEST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "best_{}".format(CHECKPOINT_TEMPLATE))
PNG_DPI = 200
TRAINING_LOSS_PNG_TEMPLATE = os.path.join(OUTPUT_FOLDER, "training_loss_{}.png")
TEST_LOSS_PNG_FORMAT = os.path.join(OUTPUT_FOLDER, "loss_histogram_{}.png")
SINGLE_EVENT_LOSS_FILE_TEMPLATE = os.path.join(OUTPUT_FOLDER, "losses_{}.npy")


class BasicAutoEncoder(nn.Module):

    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, dropout=DROPOUT):
        """
        @param input_dim: the amount of data points to take per event
        @param latent_dim: the dimension of the latent space, this is the effective dimension of the input after
                           training
        """
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

    def __init__(self, filename, chunk_size, total_size, input_dim, batch_size, shuffle):
        self.filename = filename
        self.chunk_size = chunk_size
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
            labels = x.rename_axis('ID').values[:, -1]
            x = x.rename_axis('ID').values[:, :self.input_dim]
            loader = data.DataLoader(x, batch_size=self.batch_size, shuffle=self.shuffle)
            self.idx += 1
            return loader, labels.tolist()

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.total_size / self.chunk_size))


def train(net, optimizer, data_loader_gen, criterion, epochs, last_cp_path, best_cp_path, lr):
    print("Training for {} epochs".format(epochs))
    net.train()
    losses = []
    epochs_since_last_improvement = 0
    for epoch_num in tqdm(range(epochs)):
        if epochs_since_last_improvement > PATIENCE:
            print("Terminating due to impatience. No major improvements seen in"
                  " the last {} epochs.".format(epochs_since_last_improvement))
            return losses
        data_loader_gen.reset()
        epoch_losses = []
        for data_loader, _ in data_loader_gen:
            mini_epoch_loss = 0
            for x in data_loader:
                optimizer.zero_grad()  # zero the gradient buffers
                output = net(x.float())
                loss = criterion(output, x.float())
                loss.backward()
                optimizer.step()  # Does the update
                mini_epoch_loss += loss.detach().item()
                gc.collect()
            epoch_losses.append(mini_epoch_loss)
        losses += epoch_losses
        if save(net, optimizer, sum(epoch_losses), epoch_num, last_cp_path, best_cp_path, lr):
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
    return losses


def save(net, optimizer, loss, epoch, path, best_path=None, lr=None):
    print("Saving checkpoints for epoch: {}".format(epoch))
    cp_dict = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(cp_dict, path)
    if not os.path.exists(best_path):
        print("Best checkpoint not found.\nInitializing best checkpoint.")
        torch.save(cp_dict, best_path)
    elif best_path and lr:
        _, _, _, best_loss = load(best_path, lr)
        if (loss / best_loss) < LOSS_IMPROVEMENT_THRESHOLD:
            print("Epoch {} is less than {} smaller than the previous best."
                  "\nUpdating best checkpoint.".format(epoch, LOSS_IMPROVEMENT_THRESHOLD))
            torch.save(cp_dict, best_path)
            return True
    return False


def load(path, lr):
    checkpoint = torch.load(path)
    net = BasicAutoEncoder()
    optimizer = optim.AdamW(net.parameters(), lr)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optimizer, epoch, loss


def test(net, data_loader_gen, criterion, name):
    print("Starting test for: {}".format(name))
    loss_file_name = SINGLE_EVENT_LOSS_FILE_TEMPLATE.format(name)
    if os.path.exists(loss_file_name):
        print("Previous loss file detected, loading losses")
        losses = np.load(loss_file_name, "r+")
    else:
        print("No previous loss file detected, calculating losses")
        net.eval()
        with torch.no_grad():
            losses = [[], []]  # first column is loss and second is label
            data_loader_gen.reset()
            data_loader_gen.batch_size = 1
            data_loader_gen.shuffle = False
            data_loader_gen.chunk_size = 10**3
            for data_loader, labels in tqdm(data_loader_gen):
                losses[1] += labels
                for x in data_loader:
                    output = net(x.float())
                    loss = criterion(output, x.float())
                    losses[0].append(loss.detach().item())
                    gc.collect()
        print("Losses calculated")
        losses = np.array(losses)
        np.save(loss_file_name, losses)
        print("Losses saved to: {}".format(loss_file_name))
    print("Separating losses to sig and bg")
    sig_losses = np.array([losses[0, i] for i in tqdm(range(len(losses[0]))) if losses[1, i]])
    bg_losses = np.array([losses[0, i] for i in tqdm(range(len(losses[0]))) if not losses[1, i]])
    gc.collect()
    plot_histogram(losses, "all data loss", name, TEST_LOSS_PNG_FORMAT.format(name))
    plot_histogram(sig_losses, "sig loss", name, TEST_LOSS_PNG_FORMAT.format(name+"_sig"))
    plot_histogram(bg_losses, "bg loss", name, TEST_LOSS_PNG_FORMAT.format(name+"_bg"))


def plot_losses(losses, path, plot_function=plt.plot):
    plt.figure()
    plot_function(losses)
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()
    print("Plotted: {}".format(path))


def plot_histogram(data_set, x_axis_label,  name, path, nbins=NBINS, trim_percent=TRIM_PERCENT):
    plt.figure()
    trimmed = trim_outliers(data_set, trim_percent)
    bins = np.histogram(trimmed, bins=nbins)[1]
    plt.hist(data_set, color="b", label="bg", log=True, bins=bins)
    plt.legend()
    plt.title(name + " {} histogram".format(x_axis_label))
    plt.xlabel(x_axis_label)
    plt.ylabel("Num events")
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()
    print("Plotted {} histogram and saved to:\n{}".format(x_axis_label, path))


def run_net(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, learning_rate=LEARNING_RATE, dropout=DROPOUT, do_train=True,
            do_test=True):
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
    if do_train and EPOCHS - epoch > 0:
        training_losses = train(net, optimizer, data_gen, CRITERION, EPOCHS - epoch, last_cp_name, best_cp_name,
                                learning_rate)
        plot_losses(training_losses, TRAINING_LOSS_PNG_TEMPLATE.format(name))
    if do_test:
        test(net, data_gen, CRITERION, name)


def parameter_search():
    n = np.random.random((10, 3))
    latent_dim = 2 ** 2
    for params in n:
        learning_rate = round(10 ** (-2.5 * params[0] - 2.5), ndigits=5)
        dropout = round(10 ** (-2 * params[1] - 2), ndigits=5)
        input_dim = 2 ** (int(2 * params[2]) + 6)  # 64 or 128
        print("lr={}, do={}, input={}".format(learning_rate, dropout, input_dim))
        run_net(input_dim, latent_dim, learning_rate, dropout)


def main():
    # parameter_search()
    run_net(do_train=False)
    run_net(input_dim=2**6)


if __name__ == "__main__":
    main()

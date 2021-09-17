import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import gc
from copy import deepcopy
from auto_encoder_dataloader import DataLoaderGenerator
from auto_encoder_model import BasicAutoEncoder
from auto_encoder_losses import Losses

"""
Parameter search results:
input layer = 128 or 64
latent layer = 4
learning rate = 0.002
dropout = 0.001
Running for 400 epochs to see if we can get good results
"""

RUN = "cluster_raw"
PARAM_DICT = {
    "rotemov_raw": (
    "C:\\Users\\rotem\\PycharmProjects\\ML4Jets\\ML4Jets-HUJI\\Data\\events_anomalydetection.h5", 10 ** 3,
    10 ** 4, 3),
    "rotemov_obs": ("external_maf/datasets/data/lhc/lhc.npy", 10 ** 3, 2 * 10 ** 3, 2),
    "cluster_raw": (
    "/usr/people/snirgaz/rotemov/rotemov/Projects/ML4Jets-HUJI/Data/events_anomalydetection.h5", 10 ** 5,
    int(1.1 * 10 ** 6), 400),
    "cluster_obs": ("external_maf/datasets/data/lhc/lhc_R0.4_all.npy", int(1.1 * 10 ** 6), int(1.1 * 10 ** 6), 30),
}

DATA_SET_PATH, MINI_EPOCH_SIZE, EPOCH_SIZE, EPOCHS = PARAM_DICT[RUN]
BATCH_SIZE = int(2.5 * 10 ** 2)

CRITERION = Losses.mse_k_means  # the loss function
SHUFFLE = True
LEARNING_RATE = 0.002
DROPOUT = 0.001
INPUT_DIM = 128
LATENT_DIM = 4
LOSS_IMPROVEMENT_THRESHOLD = 0.999
PATIENCE = 30
TRIM_PERCENT = 0.02
NBINS = 100

OUTPUT_FOLDER = "ae_models"
NAME_TEMPLATE = "in{}_lat{}_lr{}_do{}"
CHECKPOINT_TEMPLATE = "basic_ae_checkpoint_{}.pt"  # input size, latent size, learning rate, dropout
LAST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "last_{}".format(CHECKPOINT_TEMPLATE))
BEST_CHECKPOINT_PATH_TEMPLATE = os.path.join(OUTPUT_FOLDER, "best_{}".format(CHECKPOINT_TEMPLATE))
PNG_DPI = 200
TRAINING_LOSS_PNG_TEMPLATE = os.path.join(OUTPUT_FOLDER, "training_loss_{}.png")
TEST_LOSS_PNG_FORMAT = os.path.join(OUTPUT_FOLDER, "loss_histogram_{}.png")
SINGLE_EVENT_LOSS_FILE_TEMPLATE = os.path.join(OUTPUT_FOLDER, "losses_{}.npy")


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
            for x in tqdm(data_loader):
                optimizer.zero_grad()  # zero the gradient buffers
                output, latent = net(x.float())
                loss = criterion(output, x.float(), latent, epoch_num)
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
        'centroids': Losses.CENTROIDS
    }
    torch.save(cp_dict, path)
    if not os.path.exists(best_path):
        print("Best checkpoint not found.\nInitializing best checkpoint.")
        torch.save(cp_dict, best_path)
    elif best_path and lr:
        _, _, _, best_loss = load(best_path, lr, net.encoder_layer_sizes, net.decoder_layer_sizes, net.dropout)
        if (loss / best_loss) < LOSS_IMPROVEMENT_THRESHOLD:
            print("Epoch {} is less than {} smaller than the previous best."
                  "\nUpdating best checkpoint.".format(epoch, LOSS_IMPROVEMENT_THRESHOLD))
            torch.save(cp_dict, best_path)
            return True
    return False


def load(path, lr, encoder_layer_sizes, decoder_layer_sizes, dropout):
    checkpoint = torch.load(path)
    net = BasicAutoEncoder(encoder_layer_sizes, decoder_layer_sizes, dropout=dropout)
    optimizer = optim.AdamW(net.parameters(), lr)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    Losses.CENTROIDS = checkpoint['centroids']
    return net, optimizer, epoch, loss


def test(net, data_loader_gen, criterion, name):
    print("Starting test for: {}".format(name))
    loss_file_name = SINGLE_EVENT_LOSS_FILE_TEMPLATE.format(name)
    if os.path.exists(loss_file_name):
        print("Previous loss file detected, loading losses")
        losses = np.load(loss_file_name, "r")
    else:
        print("No previous loss file detected, calculating losses")
        net.eval()
        with torch.no_grad():
            losses = [[], []]  # first column is loss and second is label
            data_loader_gen.reset()
            data_loader_gen.batch_size = 1
            data_loader_gen.shuffle = False
            data_loader_gen.chunk_size = 10 ** 3
            for data_loader, labels in tqdm(data_loader_gen):
                losses[1] += labels
                for x in tqdm(data_loader):
                    output, latent = net(x.float())
                    loss = criterion(output, x.float(), latent, iteration=0)
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
    plot_histogram(losses[0, :], "all data loss", name, TEST_LOSS_PNG_FORMAT.format(name))
    plot_histogram(sig_losses, "sig loss", name, TEST_LOSS_PNG_FORMAT.format(name + "_sig"))
    plot_histogram(bg_losses, "bg loss", name, TEST_LOSS_PNG_FORMAT.format(name + "_bg"))


def plot_losses(losses, path):
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()
    print("Plotted: {}".format(path))


def trim_outliers(data, trim_percent):
    """
    @param data: A one dimensional array
    @param trim_percent: The percentage to cut off the edges
    """
    data_size = data.shape[0]
    start = int(data_size * trim_percent)
    end = data_size - start
    return np.sort(data)[start:end]


def plot_histogram(data_set, x_axis_label, name, path, nbins=NBINS, trim_percent=TRIM_PERCENT):
    plt.figure()
    trimmed = trim_outliers(data_set, trim_percent)
    bins = np.histogram(trimmed, bins=nbins)[1]
    plt.hist(data_set, label="bg", log=True, bins=bins)
    plt.legend()
    plt.title(name + " {} histogram".format(x_axis_label))
    plt.xlabel(x_axis_label)
    plt.ylabel("Num events")
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()
    print("Plotted {} histogram and saved to:\n{}".format(x_axis_label, path))


def run_net(encoder_layer_sizes, decoder_layer_sizes, learning_rate=LEARNING_RATE, dropout=DROPOUT, do_train=True,
            do_test=True):
    input_dim = encoder_layer_sizes[0]
    latent_dim = encoder_layer_sizes[-1]
    name = NAME_TEMPLATE.format(input_dim, latent_dim, learning_rate, dropout)
    last_cp_name = LAST_CHECKPOINT_PATH_TEMPLATE.format(name)
    best_cp_name = BEST_CHECKPOINT_PATH_TEMPLATE.format(name)
    if os.path.exists(last_cp_name):
        net, optimizer, epoch, _ = load(last_cp_name, learning_rate, encoder_layer_sizes, decoder_layer_sizes, dropout)
    else:
        net = BasicAutoEncoder(encoder_layer_sizes, decoder_layer_sizes, dropout)
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
        epoch = 0
    data_gen = DataLoaderGenerator(DATA_SET_PATH, MINI_EPOCH_SIZE, EPOCH_SIZE, input_dim, BATCH_SIZE, SHUFFLE)
    if do_train and EPOCHS - epoch > 0:
        training_losses = train(net, optimizer, data_gen, CRITERION, EPOCHS - epoch, last_cp_name, best_cp_name,
                                learning_rate)
        plot_losses(training_losses, TRAINING_LOSS_PNG_TEMPLATE.format(name))
    if do_test:
        test(net, data_gen, CRITERION, name)


def parameter_search(encoder_layer_sizes, decoder_layer_sizes, experiment_name):
    # TODO: Apply to new network config
    encoder_layer_sizes = encoder_layer_sizes
    decoder_layer_sizes = decoder_layer_sizes
    n = np.random.random((10, 3))
    for params in n:
        learning_rate = round(10 ** (-2.5 * params[0] - 2.5), ndigits=5)  # between 10^-2.5 to 10^-5
        dropout = round(10 ** (-3 * params[1] - 2), ndigits=5)  # between 10^-2 to 10^-5
        latent_dim = (int(4 * params[2]) + 2)  # between 2 to 6
        enc = deepcopy(encoder_layer_sizes)
        enc[-1] = latent_dim
        print("lr={}, do={}, input={}".format(learning_rate, dropout, latent_dim))
        print(enc)
        run_net(enc, decoder_layer_sizes, learning_rate, dropout, do_test=False)

    # moving png to folder
    layer_string = "_".join([str(x) for x in encoder_layer_sizes])
    full_name = "{}_{}".format(experiment_name, layer_string)
    experiment_folder = os.path.join(OUTPUT_FOLDER, full_name)
    print(os.system("mkdir {}".format(experiment_folder)))
    print(os.system("mv {}/*.png {}/.".format(OUTPUT_FOLDER, experiment_folder)))


def main():
    parameter_search([11, 128, 64, 32, 16, 8, 4], [8, 16, 32, 64, 128, 11], "deep_wide")
    # parameter_search([11, 128, 4], [128, 11], "shallow_wide")
    # parameter_search([11, 32, 64, 128, 64, 32, 16, 8, 4], [8, 16, 32, 64, 128, 64, 32, 11], "deep_wide")
    # parameter_search([11, 7, 4], [7, 11], "shallow_narrow")
    # run_net(do_train=False)
    # run_net(input_dim=2**6)
    # run_net(encoder_layer_sizes=[6, 4, 2], decoder_layer_sizes=[4, 6], do_test=False)


if __name__ == "__main__":
    main()

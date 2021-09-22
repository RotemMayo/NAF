import numpy as np
import matplotlib.pyplot as plt
import datasets
import util


class LHC_BINNED:

    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, val_percent=0.1, test_percent=0.1, experiment_name="forgot_experiment_name", shuffle=True):
        data_path = '{}lhc/lhc_mjj_bin_{}.npy'.format(datasets.ROOT, experiment_name)
        trn, val, tst = load_data(data_path, val_percent, test_percent, shuffle)
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split, vars):
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x[:, vars])
        plt.show()


def load_data(data_path, val_percent, test_percent, shuffle=True):
    """
    signal percent must be less than 10%
    """
    np.random.seed(42)
    data = np.nan_to_num(np.load(data_path, allow_pickle=True))

    if shuffle:
        np.random.shuffle(data)

    N_test = int(test_percent*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(val_percent*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]
    return data_train, data_validate, data_test


def load_data_normalised(data_path, val_percent, test_percent):
    data_train, data_validate, data_test = load_data(data_path, val_percent, test_percent)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s
    return data_train, data_validate, data_test
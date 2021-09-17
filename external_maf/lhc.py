import numpy as np
import matplotlib.pyplot as plt
import datasets
import util


class LHC:

    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, signal_percent=0, val_percent=0.1, test_percent=0.1, experiment_name=""):
        bg = '{}lhc/bg_{}.npy'.format(datasets.ROOT, experiment_name)
        sig = '{}lhc/sig_{}.npy'.format(datasets.ROOT, experiment_name)
        trn, val, tst = load_data_normalised(bg, sig, signal_percent, val_percent, test_percent)
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


def load_data(bg_path, sig_path, signal_percent, val_percent, test_percent):
    """
    signal percent must be less than 10%
    """
    sig = np.nan_to_num(np.load(sig_path))
    bg = np.nan_to_num(np.load(bg_path))
    bg_rows, _ = bg.shape
    sig_rows, _ = sig.shape
    idx = np.random.randint(sig_rows, size=int(signal_percent*bg_rows))
    data = np.concatenate((bg, sig[idx, :]), axis=0)
    np.random.shuffle(data)

    N_test = int(test_percent*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(val_percent*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]
    return data_train, data_validate, data_test


def load_data_normalised(bg, sig, signal_percent, val_percent, test_percent):
    data_train, data_validate, data_test = load_data(bg, sig, signal_percent, val_percent, test_percent)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s
    return data_train, data_validate, data_test
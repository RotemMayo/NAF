import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class NumpyDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, data, normalize=True):
        """
        param data: a numpy array
        """
        self.length = data.shape[0]
        self.original_data = np.nan_to_num(data).astype(np.float32)
        if normalize:
            n_features = self.original_data.shape[1]
            self.mean = self.original_data.mean(axis=0).reshape(1, n_features)
            self.std = self.original_data.std(axis=0).reshape(1, n_features)
            self.data = torch.Tensor((self.original_data - self.mean) / self.std)
        else:
            self.data = torch.Tensor(self.original_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]


def get_mjj(df):
    e1sq = df['pxj1'] ** 2 + df['pyj1'] ** 2 + df['pzj1'] ** 2 + df['mj1'] ** 2
    e2sq = df['pxj2'] ** 2 + df['pyj2'] ** 2 + df['pzj2'] ** 2 + df['mj2'] ** 2
    ptotsq = (df['pxj1'] + df['pxj2']) ** 2 + (df['pyj1'] + df['pyj2']) ** 2 + (df['pzj1'] + df['pzj2']) ** 2
    mjjsq = e1sq + e2sq - ptotsq
    mjjsq[mjjsq < 0] = 0
    return mjjsq ** 0.5


def plot_mj(df, cut1, cut2, bins):
    plt.figure()
    mj = pd.concat([df['mj1'], df['mj2']], axis=1)
    plt.hist2d(x=mj.max(axis=1)[cut1], y=mj.min(axis=1)[cut1], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.figure()
    plt.hist2d(x=mj.max(axis=1)[cut2], y=mj.min(axis=1)[cut2], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()


def get_n_signal_dataset(n_sig, df):
    np.random.seed(42)
    signal = df['label'] == 1
    indices = np.random.choice(df.index[signal], n_sig, replace=False)
    mask = np.full(df.shape[0], False)
    mask[df.index[~signal]] = True
    mask[indices] = True
    return df[mask].iloc[:, :-3].to_numpy(dtype=np.float32), mask


def plot_cuts(anomally_score, bin_mask, relevant_signal, steps=100):
    signal_percent = []
    percentiles = np.linspace(0, 1, steps+1)[:-1]
    for p in tqdm(percentiles):
        cut = np.logical_and(bin_mask, anomally_score >= anomally_score[bin_mask].quantile(p))
        signal_percent.append(np.sum(1.0*np.logical_and(cut, relevant_signal))/np.sum(cut))
    base_line = np.sum(1.0*np.logical_and(bin_mask, relevant_signal))/np.sum(bin_mask)
    plt.figure()
    plt.plot(percentiles, signal_percent)
    plt.hlines(base_line, 0, 1)
    print("Cut quantile: {} \n"
          "Signal percent: {}\n"
          "Bin signal percent: {}".format(percentiles[np.argmax(signal_percent)], np.max(signal_percent), base_line))
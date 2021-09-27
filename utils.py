import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import FastICA


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


def plot_params(param1, param2, cut1, cut2, bins):
    plt.figure()
    plt.hist2d(x=param1[cut1], y=param2[cut1], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.figure()
    plt.hist2d(x=param1[cut2], y=param2[cut2], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()


def plot_mj(df, cut1, cut2, bins):
    mj = pd.concat([df['mj1'], df['mj2']], axis=1)
    m1 = mj.max(axis=1)
    m2 = mj.min(axis=1)
    plot_params(m1, m2, cut1, cut2, bins)


def get_n_signal_dataset(n_sig, df):
    np.random.seed(42)
    signal = df['label'] == 1
    indices = np.random.choice(df.index[signal], n_sig, replace=False)
    mask = np.full(df.shape[0], False)
    mask[df.index[~signal]] = True
    mask[indices] = True
    return df[mask].to_numpy(dtype=np.float32), mask


def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_white = np.dot(U, Vt)
    return X_white


def preprocess_dataframe(df):
    X_norm = df.to_numpy()
    X_norm = (X_norm - X_norm.mean(axis=0))/X_norm.std(axis=0)
    X_norm = svd_whiten(X_norm)
    ica = FastICA(random_state=42)
    ica.fit(X_norm)
    X_norm = ica.transform(X_norm)
    print(ica.components_)
    X_norm = (X_norm - X_norm.mean(axis=0))/X_norm.std(axis=0)
    return X_norm


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

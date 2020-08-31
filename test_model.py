from maf_experiments import model
from maf_experiments import parse_args
import os
import json
import external_maf.datasets as datasets
import numpy as np
import torch.utils.data as data
import external_maf.lhc as lhc
from torch.autograd import Variable
import torch


def load_model(fn, save_dir="models"):
    args = parse_args()
    old_args = save_dir + '/' + fn + '_args.txt'
    old_path = save_dir + '/' + fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}

        d = without_keys(json.loads(open(old_args, 'r').read()), ['to_train', 'epoch'])
        args.__dict__.update(d)

        """
        if overwrite_args:
            fn = args2fn(args)
        print(" New args:")
        print(args)
        """

        print('\nfilename: ', fn)
        mdl = model(args, fn)
        print(" [*] Loading model!")
        mdl.load(old_path)
        return mdl


def load_for_test(signal_percent):
    bg_path = datasets.root + 'lhc/bg.npy'
    sig_path = datasets.root + 'lhc/sig.npy'
    sig = np.nan_to_num(np.load(sig_path))
    bg = np.nan_to_num(np.load(bg_path))
    bg_rows, _ = bg.shape
    sig_rows, _ = sig.shape
    idx = np.random.randint(sig_rows, size=int(signal_percent * bg_rows))
    data = np.concatenate((bg, sig[idx, :]), axis=0)
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    bg = (bg - mu) / s
    sig = (sig - mu) / s
    return bg, sig


def get_scores(mdl, dataset):
    size = dataset.shape[0]
    loader = data.DataLoader(lhc.LHC.Data(dataset).x, batch_size=size, shuffle=False)
    losses = np.array([])
    for x in loader:
        x = Variable(x)
        losses = mdl.maf.loss(x).data.cpu().numpy()
    return losses.reshape(size, 1)


def main():
    file_name = "lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best"
    mdl = load_model(file_name)
    bg, sig = load_for_test(mdl.args.signal_percent)

    n_bg = bg.shape[0]
    bg_scores = get_scores(mdl, bg)
    bg = np.append(bg_scores, bg, axis=1)
    bg = np.append(bg, np.zeros((n_bg, 1)), axis=1)

    n_sig = sig.shape[0]
    sig_scores = get_scores(mdl, sig)
    sig = np.append(sig_scores, sig, axis=1)
    sig = np.append(sig, np.ones((n_sig, 1)), axis=1)

    data = np.append(sig, bg, axis=0)
    sorted = data[(data[:, 0]).argsort()]

    print("Going by smallest loss")
    for i in range(6):
        n = 10**i
        print("Number of signal in top events: [" + str(int(np.sum(sorted[:n, -1]))) + "/" + str(n) + "]")

    for i in range(sorted.shape[0]-1):
        if sorted[i, 0] < sorted[i + 1, 0]:
            print("Not sorted")





if __name__ == "__main__":
    main()

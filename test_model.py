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
    scores = losses / size
    return scores



def main():
    file_name = "lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best"
    mdl = load_model(file_name)
    bg, sig = load_for_test(mdl.args.signal_percent)
    bg_scores = get_scores(mdl, bg)
    sig_score = get_scores(mdl, sig)
    bg37_score = get_scores(mdl, bg[37, :])
    print(bg_scores[37], bg37_score)


if __name__ == "__main__":
    main()

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


def get_probs(mdl, dataset):
    loader = data.DataLoader(lhc.LHC.Data(dataset).x, batch_size=mdl.args.batch_size, shuffle=False)
    p = len(dataset[0, :])
    probs = []
    for x in loader:
        x = Variable(x)
        n = x.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_())
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(n, p).zero_())
        z, logdet, _ = mdl.maf.flow((x, lgd, context))
        det = torch.exp(logdet)
        probs_tens = torch.matmul(det, z)
        print("z", z)
        print("det", det)
        print("probs", probs_tens)


def main():
    file_name = "lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best"
    mdl = load_model(file_name)
    print(mdl.args, mdl.checkpoint)
    bg, sig = load_for_test(mdl.args.signal_percent)
    print(bg.shape, sig.shape, bg[1:3, :], sig[1:3, :])
    get_probs(mdl, bg[0:10, :])


if __name__ == "__main__":
    main()

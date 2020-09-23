from maf_experiments import model
from maf_experiments import parse_args
import os
import subprocess
import json
import external_maf.datasets as datasets
import numpy as np
import torch.utils.data as data
import external_maf.lhc as lhc
from torch.autograd import Variable
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_FILE = "results/output.txt"
PDF_NAME_FORMAT = "results/{}_loss_plots.pdf"
OBS_LABELS = ["Loss", "M_{jj}", "N_{j}", "m_{1}", "m_{2}", "First jet {\Tau}_{21}",
              "Second jet {\Tau}_{21}", "Classifier"]
FILES_TO_TEST = [
    ("lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0, "affine"),
    ("lhc_sp0.001_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.001, "affine"),
    ("lhc_sp0.01_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.01, "affine"),
    ("lhc_sp0.025_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.025, "affine"),
    ("lhc_sp0.05_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.05, "affine"),
    ("lhc_sp0.075_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.075, "affine"),
    ("lhc_sp0.1_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.1, "affine"),
    ("lhc_sp0.001_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.001, "ddsf"),
    ("lhc_sp0.01_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.01, "ddsf"),
    ("lhc_sp0.025_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.025, "ddsf"),
    ("lhc_sp0.05_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.05, "ddsf"),
    ("lhc_sp0.075_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.075, "ddsf"),
    ("lhc_sp0.1_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.1, "ddsf"),
]


def load_model(fn, save_dir="models"):
    args = parse_args()
    old_args = save_dir + '/' + fn + '_args.txt'
    old_path = save_dir + '/' + fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(json.loads(open(old_args, 'r').read()), ['to_train', 'epoch'])
        args.__dict__.update(d)
        print_to_file('\nfilename: ' + fn)
        mdl = model(args, fn)
        print_to_file(" [*] Loading model!")
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


def print_to_file(msg):
    # print(msg)
    # subprocess.call(["echo ", msg], shell=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg)


def all_plots(sig, bg, name):
    with PdfPages(name) as pdf:
        sig_loss = sig[:, 0]
        bg_loss = bg[:, 0]

        # Plotting histograms
        plt.figure()
        plt.hist(sig_loss, color="r", label="sig")
        plt.hist(bg_loss, color="b", label="bg")
        plt.legend()
        plt.xlabel(OBS_LABELS[0])
        plt.ylabel("Num events")
        pdf.savefig()
        plt.close()

        # Plotting observables vs loss
        for i in range(1, sig.shape[1] - 2):
            sig_obs = sig[:, i]
            bg_obs = bg[:, i]
            plt.figure()
            plt.plot(sig_obs, sig_loss, color="r", label="sig", marker='.')
            plt.plot(bg_obs, bg_loss, color="b", label="bg", marker='.')
            plt.legend()
            plt.xlabel(OBS_LABELS[i])
            plt.ylabel(OBS_LABELS[0])
            pdf.savefig()
            plt.close()


def test_model(file_name, sp, flow_type):
    print_to_file("Signal percent: " + str(sp*100))
    print_to_file("Num signals: " + str(sp*10**5))
    print_to_file("Num bg: " + str(10**6))
    print_to_file("Flow type: " + flow_type)
    print_to_file("File name: " + file_name)
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
    sorted = data[(-data[:, 0]).argsort()]

    print_to_file("Going by largest loss: ")
    for i in range(7):
        n = 10**i
        print_to_file("Number of signal in top events: [" + str(int(np.sum(sorted[:n, -1]))) + "/" + str(n) + "]")

    print_to_file("Going by smallest loss: ")
    for i in range(7):
        n = 10 ** i
        print_to_file("Number of signal in bottom events: [" + str(int(np.sum(sorted[-n:, -1]))) + "/" + str(n) + "]")
    print_to_file("=========================================================================\n\n")
    all_plots(sig, bg, PDF_NAME_FORMAT.format(file_name))


def main():
    for file_data in FILES_TO_TEST:
        test_model(*file_data)


if __name__ == "__main__":
    main()

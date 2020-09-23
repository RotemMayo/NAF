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
from tqdm import tqdm
from datetime import datetime
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

PLOT_FLAG = True
PDF_FLAG = False
PNG_FLAG = True
OBS_LABELS = ["Loss", "Mjj", "Nj", "Mtot", "m1", "m2", "First_jet_tau21",
              "Second_jet_tau_21", "Classifier"]
FILES_TO_TEST = [
    ("lhc_sp0.1_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.1, "ddsf"),
    ("lhc_sp0.01_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.01, "ddsf"),
    ("lhc_sp0.025_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.025, "ddsf"),
    ("lhc_sp0.05_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.05, "ddsf"),
    ("lhc_sp0.075_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.075, "ddsf"),
    ("lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0, "affine"),
    ("lhc_sp0.001_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.001, "affine"),
    ("lhc_sp0.01_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.01, "affine"),
    ("lhc_sp0.025_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.025, "affine"),
    ("lhc_sp0.05_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.05, "affine"),
    ("lhc_sp0.075_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.075, "affine"),
    ("lhc_sp0.1_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best", 0.1, "affine"),

]
NUMBERS_TO_CHECK = [10**j for j in range(7)] + [j*10**4 for j in range(1, 10)] + [j*10**5 for j in range(1, 10)]
MIN_LOSS = -5
MAX_LOSS = 100
NBINS = 300
TIME_STAMP = datetime.now().strftime("%d%m%Y_%H%M%S")
RUN_OUTPUT_DIR = "results/run_{}/".format(TIME_STAMP)
OUTPUT_FILE = "{}all_results.txt".format(RUN_OUTPUT_DIR)
MODEL_OUTPUT_DIR_FORMAT = RUN_OUTPUT_DIR + "{}/"
PDF_NAME_FORMAT = "{}plots.pdf"
PNG_NAME_FORMAT = "{}{}.png"


def load_model(fn, save_dir="models"):
    args = parse_args()
    old_args = save_dir + '/' + fn + '_args.txt'
    old_path = save_dir + '/' + fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(json.loads(open(old_args, 'r').read()), ['to_train', 'epoch'])
        args.__dict__.update(d)
        # print_to_file('\nfilename: ' + fn)
        mdl = model(args, fn)
        # print_to_file(" [*] Loading model!")
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
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")
    # print(msg)
    # subprocess.call(["echo ", msg], shell=True)


def save_plot(pdf, png_path):
    if PDF_FLAG:
        pdf.savefig()
    if PNG_FLAG:
        plt.savefig(png_path, dpi=300)


def all_plots(sig, bg, name):
    output_dir = MODEL_OUTPUT_DIR_FORMAT.format(name)
    if not os.path.isdir(output_dir):
        os.mkdir(MODEL_OUTPUT_DIR_FORMAT.format(name))
    pdf_path = PDF_NAME_FORMAT.format(output_dir)

    with PdfPages(pdf_path) as pdf:
        sig_loss = sig[:, 0]
        bg_loss = bg[:, 0]

        # Plotting histograms
        plt.figure()
        bins = np.histogram(np.hstack((sig_loss, bg_loss)), bins=NBINS)[1]
        plt.hist(bg_loss, color="b", label="bg", log=True, bins=bins)
        plt.hist(sig_loss, color="r", label="sig", log=True, bins=bins)
        plt.legend()
        plt.xlabel(OBS_LABELS[0])
        plt.ylabel("Num events")
        png_path = PNG_NAME_FORMAT.format(output_dir, "histogram")
        save_plot(pdf, png_path)
        plt.xlim(MIN_LOSS, MAX_LOSS)
        png_path = PNG_NAME_FORMAT.format(output_dir, "histogram_no_outliers")
        save_plot(pdf, png_path)
        plt.close()


        # Plotting observables vs loss
        for i in tqdm(range(1, sig.shape[1] - 2)):
            sig_obs = sig[:, i]
            bg_obs = bg[:, i]
            plt.figure()
            plt.scatter(bg_obs, bg_loss, color="b", label="bg", marker='.')
            plt.scatter(sig_obs, sig_loss, color="r", label="sig", marker='.')
            plt.legend()
            plt.xlabel(OBS_LABELS[i])
            plt.ylabel(OBS_LABELS[0])
            png_path = PNG_NAME_FORMAT.format(output_dir, OBS_LABELS[i])
            save_plot(pdf, png_path)
            plt.close()


def test_model(file_name, sp, flow_type):
    print_to_file("Signal percent: " + str(sp * 100))
    print_to_file("Num signals: " + str(sp * 10 ** 5))
    print_to_file("Num bg: " + str(10 ** 6))
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
    for n in NUMBERS_TO_CHECK:
        print_to_file("Number of signal in top events: [" + str(int(np.sum(sorted[:n, -1]))) + "/" + str(n) + "]")

    print_to_file("Going by smallest loss: ")
    for n in NUMBERS_TO_CHECK:
        print_to_file("Number of signal in bottom events: [" + str(int(np.sum(sorted[-n:, -1]))) + "/" + str(n) + "]")
    print_to_file("=========================================================================\n\n")
    if PLOT_FLAG:
        name = flow_type + "_" + str(sp)
        all_plots(sig, bg, name)


def main():
    if not os.path.isdir(RUN_OUTPUT_DIR):
        os.mkdir(RUN_OUTPUT_DIR)
    for i in tqdm(range(len(FILES_TO_TEST))):
        file_data = FILES_TO_TEST[i]
        test_model(*file_data)


if __name__ == "__main__":
    main()

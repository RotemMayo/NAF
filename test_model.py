from maf_experiments import model
from maf_experiments import parse_args
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import external_maf.datasets as datasets
import numpy as np
import torch.utils.data as data
import external_maf.lhc as lhc
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime
import os
from fpdf import FPDF

mpl.rcParams['agg.path.chunksize'] = 10000

PLOT_FLAG = True
PDF_FLAG = True
PNG_FLAG = True
FIRST_EXPERIMENT_OBS_LIST = ["Loss", "Mjj", "Nj", "Mtot", "m1", "m2", "First_jet_tau21",
                             "Second_jet_tau_21", "Classifier"]
SECOND_EXPERIMENT_OBS_LIST = ["Loss", "Mjj", "Nj", "Mtot", "m1", "m2", "m1 - m2", "Lead pt", "Ht", "MHt",
                              "First_jet_tau21", "Second_jet_tau_21", "Classifier"]

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

SECOND_EXPERIMENTS = {
    "all_mjj-translation_10000": SECOND_EXPERIMENT_OBS_LIST,
    "all_mjj-translation_5000": SECOND_EXPERIMENT_OBS_LIST,
    "all_mjj-translation_1000": SECOND_EXPERIMENT_OBS_LIST,
    "all_mjj-translation_100": SECOND_EXPERIMENT_OBS_LIST,
    "all": SECOND_EXPERIMENT_OBS_LIST,
    "mjj": ["Loss", "mjj", "Classifier"],
    "mjj_m1": ["Loss", "mjj", "m1", "Classifier"],
    "mjj_m1_m1minusm2": ["Loss", "mjj", "m1", "m1 - m2", "Classifier"],
    "mjj_m1minusm2": ["Loss", "mjj", "m1 - m2", "Classifier"],
    "anode": ["Loss", "m1", "m1_minus_m2", "tau21_1", "tau21_2", "Classifier"],
    "salad": ["Loss", "m1", "m2", "First tau21", "Second tau21", "Classifier"],
}

SECOND_EXPERIMENT_R_VALUES = 1.0, 0.4
SECOND_EXPERIMENTS_NAME_FORMAT = "R{}_{}"
SECOND_EXPERIMENTS_FILE_FORMAT = "lhc_en{}_sp0.1_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_cudaFalse_best"
for en in SECOND_EXPERIMENTS.keys():
    for R in SECOND_EXPERIMENT_R_VALUES:
        full_experiment_name = SECOND_EXPERIMENTS_NAME_FORMAT.format(R, en)
        FILES_TO_TEST += [(SECOND_EXPERIMENTS_FILE_FORMAT.format(full_experiment_name), 0.1, "affine",
                           full_experiment_name, SECOND_EXPERIMENTS[en])]

NUMBERS_TO_CHECK = [10 ** j for j in range(7)] + [j * 10 ** 4 for j in range(1, 10)] + [j * 10 ** 5 for j in range(1, 10)]
MIN_LOSS = -5
MAX_LOSS = 100
NBINS = 300
TIME_STAMP = datetime.now().strftime("%d%m%Y_%H%M%S")
RUN_OUTPUT_DIR = "results/run_{}/".format(TIME_STAMP)
OUTPUT_FILE = "{}all_results.txt".format(RUN_OUTPUT_DIR)
MODEL_OUTPUT_DIR_FORMAT = RUN_OUTPUT_DIR + "{}/"
PDF_PATH_FORMAT = RUN_OUTPUT_DIR + "{}.pdf"
PNG_NAME_FORMAT = "{}{}.png"
PNG_DPI = 100
SCATTER_ALPHA = 0.5


def load_model(fn, save_dir="models"):
    args = parse_args()
    old_args = save_dir + '/' + fn + '_args.txt'
    old_path = save_dir + '/' + fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}

        d = without_keys(json.loads(open(old_args, 'r').read()), ['to_train', 'epoch'])
        args.__dict__.update(d)
        mdl = model(args, fn)
        mdl.load(old_path)
        return mdl


def normalize_data(signal_percent, bg, sig):
    bg_rows, _ = bg.shape
    sig_rows, _ = sig.shape
    idx = np.random.randint(sig_rows, size=int(signal_percent * bg_rows))
    data = np.concatenate((bg, sig[idx, :]), axis=0)
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    bg_norm = (bg - mu) / s
    sig_norm = (sig - mu) / s
    return bg_norm, sig_norm


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


def save_plot(png_path):
    if PNG_FLAG:
        plt.savefig(png_path, dpi=PNG_DPI)


def create_pdf(image_dir, img_format=".png"):
    pdf_name = image_dir[:-1] + ".pdf"
    pdf = FPDF()
    image_list = [(image_dir + i) for i in os.listdir(image_dir) if i.endswith(img_format)]
    for image in image_list:
        pdf.add_page()
        pdf.image(image)
        print("Image saved: " + image)
    pdf.output(pdf_name, "F")


def all_plots(sig, bg, name, obs_list):
    output_dir = MODEL_OUTPUT_DIR_FORMAT.format(name)
    if not os.path.isdir(output_dir):
        os.mkdir(MODEL_OUTPUT_DIR_FORMAT.format(name))

    sig_loss = sig[:, 0]
    bg_loss = bg[:, 0]

    # Plotting histograms
    plt.figure()
    bins = np.histogram(np.hstack((sig_loss, bg_loss)), bins=NBINS)[1]
    plt.hist(bg_loss, color="b", label="bg", log=True, bins=bins)
    plt.hist(sig_loss, color="r", label="sig", log=True, bins=bins)
    plt.legend()
    plt.xlabel(obs_list[0])
    plt.ylabel("Num events")
    save_plot(PNG_NAME_FORMAT.format(output_dir, "histogram"))
    plt.xlim(MIN_LOSS, MAX_LOSS)
    save_plot(PNG_NAME_FORMAT.format(output_dir, "histogram_no_outliers"))
    plt.close()

    # Plotting observables vs loss
    for i in tqdm(range(1, sig.shape[1] - 1)):
        sig_obs = sig[:, i]
        bg_obs = bg[:, i]
        plt.figure()
        plt.hexbin(bg_obs, bg_loss, cmap='Blues', mincnt=1, vmax=50, alpha=SCATTER_ALPHA, linewidths=0)
        plt.hexbin(sig_obs, sig_loss, cmap='Reds', mincnt=1, vmax=50, alpha=SCATTER_ALPHA, linewidths=0)
        plt.legend()
        plt.xlabel(obs_list[i])
        plt.ylabel(obs_list[0])
        png_path = PNG_NAME_FORMAT.format(output_dir, obs_list[i])
        save_plot(png_path)
        plt.close()

    if PDF_FLAG:
        create_pdf(output_dir)


def test_model(file_name, sp, flow_type, experiment_name="", obs_list=FIRST_EXPERIMENT_OBS_LIST):
    print_to_file("Signal percent: " + str(sp * 100))
    print_to_file("Num signals: " + str(sp * 10 ** 5))
    print_to_file("Num bg: " + str(10 ** 6))
    print_to_file("Flow type: " + flow_type)
    print_to_file("File name: " + file_name)

    mdl = load_model(file_name)
    bg = np.nan_to_num(np.load('{}lhc/bg_{}.npy'.format(datasets.ROOT, experiment_name)))
    sig = np.nan_to_num(np.load('{}lhc/sig_{}.npy'.format(datasets.ROOT, experiment_name)))
    bg_norm, sig_norm = normalize_data(mdl.args.signal_percent, bg, sig)

    n_bg = bg_norm.shape[0]
    bg_scores = get_scores(mdl, bg_norm)
    bg = np.append(bg_scores, bg, axis=1)
    bg = np.append(bg, np.zeros((n_bg, 1)), axis=1)

    n_sig = sig.shape[0]
    sig_scores = get_scores(mdl, sig_norm)
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
        name = experiment_name + flow_type + "_" + str(sp)
        all_plots(sig, bg, name, obs_list)


def main():
    if not os.path.isdir(RUN_OUTPUT_DIR):
        os.mkdir(RUN_OUTPUT_DIR)
    for i in tqdm(range(len(FILES_TO_TEST))):
        file_data = FILES_TO_TEST[i]
        test_model(*file_data)


if __name__ == "__main__":
    main()

from __future__ import division
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
#from fpdf import FPDF
import sys
from sklearn.manifold import TSNE as tsne
import pandas as pd
import seaborn as sn

mpl.rcParams['agg.path.chunksize'] = 10000

PLOT_FLAG = True
PDF_FLAG = True
PNG_FLAG = True
FIRST_EXPERIMENT_OBS_LIST = ["Loss", "Mjj", "Nj", "Mtot", "m1", "m2", "First_jet_tau21",
                             "Second_jet_tau_21", "Classifier"]
SECOND_EXPERIMENT_OBS_LIST = ["Loss", "Mjj", "Nj", "Mtot", "m1", "m2", "m1 - m2", "Lead pt", "Ht", "MHt",
                              "First_jet_tau21", "Second_jet_tau_21", "Classifier"]
INTEREST_THRESHOLD = 0.03
INTEREST_NUMBER = 10 ** 5
NUM_EVENTS_TSNE = 3 * 10 ** 4

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

FILES_TO_TEST = []

SECOND_EXPERIMENTS = {
    "all_filter_1": SECOND_EXPERIMENT_OBS_LIST,
    "all_filter_2": SECOND_EXPERIMENT_OBS_LIST,
    "all_filter_3": SECOND_EXPERIMENT_OBS_LIST,
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

SECOND_EXPERIMENTS = {
    "all_filter_4": SECOND_EXPERIMENT_OBS_LIST,
    "all_filter_3": SECOND_EXPERIMENT_OBS_LIST,
    "all_mjj-translation_1000": SECOND_EXPERIMENT_OBS_LIST,
    "all": SECOND_EXPERIMENT_OBS_LIST,
}

SECOND_EXPERIMENT_R_VALUES = [0.4]  # [1.0, 0.4]
SECOND_EXPERIMENTS_SP = [0.1]  # 0.05, 0.01]
SECOND_EXPERIMENTS_NAME_FORMAT = "R{}_{}"
SECOND_EXPERIMENTS_FILE_FORMAT = "lhc_en{}_sp{}_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_cudaFalse_best"
for en in SECOND_EXPERIMENTS.keys():
    for R in SECOND_EXPERIMENT_R_VALUES:
        for sp in SECOND_EXPERIMENTS_SP:
            full_experiment_name = SECOND_EXPERIMENTS_NAME_FORMAT.format(R, en, sp)
            FILES_TO_TEST += [(SECOND_EXPERIMENTS_FILE_FORMAT.format(full_experiment_name, sp), sp, "affine",
                               full_experiment_name, SECOND_EXPERIMENTS[en])]

NUMBERS_TO_CHECK = [10 ** j for j in range(7)] + [j * 10 ** 4 for j in range(1, 10)] + [j * 10 ** 5 for j in
                                                                                        range(1, 10)]
TRIM_PERCENT = 0.02
NBINS = 3000
TIME_STAMP = datetime.now().strftime("%d%m%Y_%H%M%S")
RUN_OUTPUT_DIR = "results/run_{}/".format(TIME_STAMP)
OUTPUT_FILE = "{}all_results.txt".format(RUN_OUTPUT_DIR)
MODEL_OUTPUT_DIR_FORMAT = RUN_OUTPUT_DIR + "{}/"
PDF_PATH_FORMAT = RUN_OUTPUT_DIR + "{}.pdf"
PNG_NAME_FORMAT = "{}{}.png"
PNG_DPI = 50
SCATTER_ALPHA = 0.25
PLOT_TITLE_FORMAT = "{}: {} vs {}"

BG_LOSS_DATA_FILE_NAME = "results/losses/bg_{}_loss.npy"
SIG_LOSS_DATA_FILE_NAME = "results/losses/sig_{}_loss.npy"
TSNE_DATA_FILE_NAME = "results/tsne_data/{}_tsne_{}.csv".format("{}", NUM_EVENTS_TSNE)

DATA_FRAME_PATH = "external_maf/datasets/data/lhc/lhc_features_and_bins.csv"
DATA_FRAME_DENSITY_PATH = "external_maf/datasets/data/lhc/lhc_{}_density.csv"


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


def get_scores(mdl, normalized_dataset):
    size = normalized_dataset.shape[0]
    loader = data.DataLoader(lhc.LHC.Data(normalized_dataset).x, batch_size=size, shuffle=False)
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


def plot_tsne(bg, sig, plot_path, data_path):
    if os.path.exists(data_path):
        tsne_df = pd.read_csv(data_path)
    else:
        data_and_labels = np.concatenate((bg[:NUM_EVENTS_TSNE, :], sig[:NUM_EVENTS_TSNE, :]), axis=0)
        data = data_and_labels[:, :-1]
        labels = data_and_labels[:, -1]
        mdl = tsne(n_components=2, random_state=0)
        # configuring the parameteres
        # the number of components = 2
        # default perplexity = 30
        # default learning rate = 200
        # default Maximum number of iterations for the optimization = 1000
        tsne_data = mdl.fit_transform(data)  # creating a new data frame which help us in ploting the result data
        tsne_data = np.vstack((tsne_data.T, labels)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))  # Ploting the result of tsne
        tsne_df.to_csv(data_path)

    sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, "Dim_1", "Dim_2", alpha=SCATTER_ALPHA / 5).add_legend()

    # sn.FacetGrid(tsne_df, hue="label", size=2).map(plt.hexbin, "Dim_1", "Dim_2", mincnt=10, vmax=50, alpha=SCATTER_ALPHA*1.5, linewidths=0).add_legend()
    save_plot(plot_path)
    plt.close()


def trim_outliers(data, trim_percent):
    """
    @param data: A one dimensional array
    @param trim_percent: The percentage to cut off the edges
    """
    l = data.shape[0]
    start = int(l*trim_percent)
    end = l-start
    return np.sort(data)[start:end]


def plot_histograms(sig, bg, sig_loss, bg_loss, name, obs_list, output_dir, trim_percent=TRIM_PERCENT):
    sig[:, 0] = sig_loss
    bg[:, 0] = bg_loss
    for i in tqdm(range(0, sig.shape[1] - 1)):
        plt.figure()
        combined = trim_outliers(np.hstack((sig[:, i], bg[:, i])), trim_percent)
        bins = np.histogram(combined, bins=NBINS)[1]
        plt.hist(bg[:, i], color="b", label="bg", log=True, bins=bins)
        plt.hist(sig[:, i], color="r", label="sig", log=True, bins=bins)
        plt.legend()
        plt.title(name + " {} histogram".format(obs_list[i]))
        plt.xlabel(obs_list[i])
        plt.ylabel("Num events")
        save_plot(PNG_NAME_FORMAT.format(output_dir, "{}_histogram".format(obs_list[i])))
        plt.close()
        plt.figure()
        plt.hist(combined, bins=bins, label="combined", color="purple", log=True)
        plt.legend()
        plt.xlabel(obs_list[i])
        plt.ylabel("Num events")
        save_plot(PNG_NAME_FORMAT.format(output_dir, "{}_combined_histogram".format(obs_list[i])))


def all_plots(sig, bg, name, obs_list):
    output_dir = MODEL_OUTPUT_DIR_FORMAT.format(name)
    if not os.path.isdir(output_dir):
        os.mkdir(MODEL_OUTPUT_DIR_FORMAT.format(name))

    sig_loss = sig[:, 0] + np.abs(np.min(sig[:, 0])) + 1
    bg_loss = bg[:, 0] + np.abs(np.min(bg[:, 0])) + 1

    # Plotting t-SNE
    plot_tsne(bg, sig, PNG_NAME_FORMAT.format(output_dir, "tsne"), TSNE_DATA_FILE_NAME.format(name))

    # Plotting histograms
    plot_histograms(sig, bg, sig_loss, bg_loss, name, obs_list, output_dir)

    # Plotting observables vs loss
    for i in tqdm(range(1, sig.shape[1] - 1)):
        sig_obs = sig[:, i]
        bg_obs = bg[:, i]
        plt.figure()
        plt.title(PLOT_TITLE_FORMAT.format(name, obs_list[0], obs_list[i]))
        plt.hexbin(bg_obs, bg_loss, cmap='Blues', mincnt=10, vmax=50, alpha=SCATTER_ALPHA, linewidths=0, yscale='log')
        plt.hexbin(sig_obs, sig_loss, cmap='Reds', mincnt=10, vmax=50, alpha=SCATTER_ALPHA, linewidths=0, yscale='log')
        plt.legend()
        plt.xlabel(obs_list[i])
        plt.ylabel(obs_list[0])
        png_path = PNG_NAME_FORMAT.format(output_dir, obs_list[i])
        save_plot(png_path)
        plt.close()

    if PDF_FLAG:
        create_pdf(output_dir)


def print_cuts(sorted_events, numbers_to_check, file_name, from_end=False):
    for n in numbers_to_check:
        if from_end:
            s = -n
            e = -1
        else:
            s = 0
            e = n
        num_sig = int(np.sum(sorted_events[s:e, -1]))
        suffix = ""
        if ((num_sig / n) < INTEREST_THRESHOLD) and (n >= INTEREST_NUMBER):
            suffix = "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            print(file_name + " is  interesting")
        print_to_file("Number of signal in bottom events: [" + str(num_sig) + "/" + str(n) + "]" + suffix)


def test_model(file_name, sp, flow_type, experiment_name="", obs_list=FIRST_EXPERIMENT_OBS_LIST):
    print("Experiment: " + experiment_name)
    name = ""
    if experiment_name != "":
        name += experiment_name + "_"
    name += (flow_type + "_sp" + str(sp))

    mdl = load_model(file_name)
    bg = np.nan_to_num(np.load('{}lhc/bg_{}.npy'.format(datasets.ROOT, experiment_name)))
    sig = np.nan_to_num(np.load('{}lhc/sig_{}.npy'.format(datasets.ROOT, experiment_name)))

    bg_loss_file_name = BG_LOSS_DATA_FILE_NAME.format(name)
    sig_loss_file_name = SIG_LOSS_DATA_FILE_NAME.format(name)
    if os.path.isfile(bg_loss_file_name) and os.path.isfile(sig_loss_file_name):
        bg_scores = np.load(bg_loss_file_name)
        sig_scores = np.load(sig_loss_file_name)
    else:
        bg_norm, sig_norm = normalize_data(mdl.args.signal_percent, bg, sig)
        bg_scores = get_scores(mdl, bg_norm)
        sig_scores = get_scores(mdl, sig_norm)
        np.save(bg_loss_file_name, bg_scores)
        np.save(sig_loss_file_name, sig_scores)

    n_bg = bg.shape[0]
    bg = np.append(bg_scores, bg, axis=1)
    bg = np.append(bg, np.zeros((n_bg, 1)), axis=1)

    n_sig = sig.shape[0]
    sig = np.append(sig_scores, sig, axis=1)
    sig = np.append(sig, np.ones((n_sig, 1)), axis=1)

    data = np.append(sig, bg, axis=0)
    sorted_events = data[(-data[:, 0]).argsort()]

    numbers_to_check = [n for n in NUMBERS_TO_CHECK if n <= sorted_events.shape[0]]

    print_to_file("Experiment: " + experiment_name)
    print_to_file("File name: " + file_name)
    print_to_file("Signal percent: " + str(sp * 100))
    print_to_file("Num signals: " + str(sig.shape[0]))
    print_to_file("Num bg: " + str(bg.shape[0]))
    print_to_file("Num signals trained on: " + str(bg.shape[0] * sp))
    print_to_file("Flow type: " + flow_type)

    print_to_file("Going by largest loss: ")
    print_cuts(sorted_events, numbers_to_check, file_name, False)

    print_to_file("Going by smallest loss: ")
    print_cuts(sorted_events, numbers_to_check, file_name, True)
    print_to_file("====================================================\n\n")

    if PLOT_FLAG:
        all_plots(sig, bg, name, obs_list)


def get_dataloader(df):
    x = np.nan_to_num(df.to_numpy()[:,:-3]).astype(np.float32)
    return data.DataLoader(x, batch_size=100, shuffle=False)


def save_densities(file_names, density_file_path):
    df = pd.read_csv(DATA_FRAME_PATH)
    models = [load_model(fn) for fn in file_names]
    print("Loaded models")
    loader = get_dataloader(df)
    density = pd.DataFrame()
    for mdl, fn in zip(models, file_names):
        densities = []
        print("Getting density for: {}".format(fn))
        for x in tqdm(loader):
            x = Variable(x)
            densities.append(-mdl.maf.loss(x).data.cpu().numpy())
        density[fn] = np.reshape(densities, df.shape[0])
        #density[fn] = np.exp(densities)
        print("Got densities for: {}".format(fn))
    if os.path.exists(density_file_path):
        tmp = pd.read_csv(density_file_path)
        density = pd.concat([tmp, density], axis=1)
    density.to_csv(density_file_path, index=False)
    print("Densities saved to: {}".format(density_file_path))


def main():
    if not os.path.isdir(RUN_OUTPUT_DIR):
        os.mkdir(RUN_OUTPUT_DIR)
    for i in tqdm(range(len(FILES_TO_TEST))):
        file_data = FILES_TO_TEST[i]
        test_model(*file_data)


def binned_main(q, flow, epochs):
    file_name_template = "lhc_binned_en{}of{}_sp0_e{}_s1993_p0.0_h100_f{}_fl5_l1_dsdim16_dsl1_cudaFalse_best".format("{}", q, epochs, flow)
    save_path = DATA_FRAME_DENSITY_PATH.format(flow)
    file_names = [file_name_template.format(i) for i in range(int(q/2), q+1)]
    save_densities(file_names, save_path)


if __name__ == "__main__":
    binned_main(8, "ddsf", 40)
    #main()

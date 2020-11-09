from test_model import *

EXPERIMENT_NAME = "R0.4_all"
SP = 0.1
MODEL_NAME = "lhc_en{}_sp{}_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_cudaFalse_best".format(EXPERIMENT_NAME, SP)
FILTER_NUMBER = 2
REMOVE_SMALLEST = 3*10**5
BG_FILE_FORMAT = "{}lhc/bg_{}_filter_{}"
SIG_FILE_FORMAT = "{}lhc/sig_{}_filter_{}"


def main():
    mdl = load_model(MODEL_NAME)
    bg = np.nan_to_num(np.load('{}lhc/bg_{}.npy'.format(datasets.ROOT, EXPERIMENT_NAME)))
    sig = np.nan_to_num(np.load('{}lhc/sig_{}.npy'.format(datasets.ROOT, EXPERIMENT_NAME)))
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

    sorted = sorted[:-REMOVE_SMALLEST]

    sig = [event[1:-1] for event in sorted if event[-1] == 1]
    bg = [event[1:-1] for event in sorted if event[-1] == 0]

    np.save(BG_FILE_FORMAT.format(datasets.ROOT, EXPERIMENT_NAME, FILTER_NUMBER), bg)
    np.save(SIG_FILE_FORMAT.format(datasets.ROOT, EXPERIMENT_NAME, FILTER_NUMBER), sig)


if __name__ == "__main__":
    main()

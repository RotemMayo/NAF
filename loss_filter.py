from test_model import *

INPUT_EXPERIMENT_NAME = "R0.4_all"
SP = 0.1
MODEL_NAME = "lhc_en{}_sp{}_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_cudaFalse_best".format(INPUT_EXPERIMENT_NAME, SP)
FILTER_NUMBER = 2

REMOVE_SMALLEST = {
    1: 3*10**5,
    2: 0
}

REMOVE_LARGEST = {
    1: 0,
    2: 10**5
}

OUTPUT_BG_FILE_FORMAT = "{}lhc/bg_{}_filter_{}"
OUTPUT_SIG_FILE_FORMAT = "{}lhc/sig_{}_filter_{}"


def main():
    mdl = load_model(MODEL_NAME)
    bg = np.nan_to_num(np.load('{}lhc/bg_{}_filter_{}.npy'.format(datasets.ROOT, INPUT_EXPERIMENT_NAME, FILTER_NUMBER-1)))
    sig = np.nan_to_num(np.load('{}lhc/sig_{}_filter_{}.npy'.format(datasets.ROOT, INPUT_EXPERIMENT_NAME, FILTER_NUMBER-1)))
    bg_norm, sig_norm = normalize_data(mdl.args.signal_percent, bg, sig)
    print(bg.shape)
    print(sig.shape)
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

    sorted = sorted[REMOVE_LARGEST[FILTER_NUMBER]:-REMOVE_SMALLEST[FILTER_NUMBER], :]
    print(sorted.shape)
    sig_new = np.ndarray([event[1:-1] for event in sorted if event[-1] == 1])
    bg_new = np.ndarray([event[1:-1] for event in sorted if event[-1] == 0])
    print(sig_new.shape)
    print(bg_new.shape)
    np.save(OUTPUT_BG_FILE_FORMAT.format(datasets.ROOT, INPUT_EXPERIMENT_NAME, FILTER_NUMBER), bg_new)
    np.save(OUTPUT_SIG_FILE_FORMAT.format(datasets.ROOT, INPUT_EXPERIMENT_NAME, FILTER_NUMBER), sig_new)


if __name__ == "__main__":
    main()

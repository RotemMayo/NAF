from test_model import *

INPUT_ORIGINAL_EXPERIMENT_NAME = "R0.4_all"
FILTER_NUMBER = 2
PREV_FILTER_EXPERIMENT = "{}_filter_{}".format(INPUT_ORIGINAL_EXPERIMENT_NAME, FILTER_NUMBER-1)
SP = 0.1
MODEL_NAME = "lhc_en{}_sp{}_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_cudaFalse_best".format(PREV_FILTER_EXPERIMENT, SP)


REMOVE_SMALLEST = {
    1: 3*10**5,
    2: 1
}

REMOVE_LARGEST = {
    1: 0,
    2: 10**5
}

OUTPUT_BG_FILE_FORMAT = "{}lhc/bg_{}_filter_{}"
OUTPUT_SIG_FILE_FORMAT = "{}lhc/sig_{}_filter_{}"


def main():
    mdl = load_model(MODEL_NAME)
    bg = np.nan_to_num(np.load('{}lhc/bg_{}.npy'.format(datasets.ROOT, PREV_FILTER_EXPERIMENT)))
    sig = np.nan_to_num(np.load('{}lhc/sig_{}.npy'.format(datasets.ROOT, PREV_FILTER_EXPERIMENT)))
    bg_norm, sig_norm = normalize_data(mdl.args.signal_percent, bg, sig)
    print(bg.shape)
    print(sig.shape)
    n_bg = bg_norm.shape[0]
    bg_scores = get_scores(mdl, bg_norm)
    print(bg_scores.shape)
    bg = np.append(bg_scores, bg, axis=1)
    bg = np.append(bg, np.zeros((n_bg, 1)), axis=1)

    n_sig = sig.shape[0]
    sig_scores = get_scores(mdl, sig_norm)
    sig = np.append(sig_scores, sig, axis=1)
    sig = np.append(sig, np.ones((n_sig, 1)), axis=1)
    print(bg.shape)
    print(sig.shape)
    data = np.append(sig, bg, axis=0)
    sorted_events = data[(-data[:, 0]).argsort()]
    print(sorted_events.shape)
    sorted_events = sorted_events[REMOVE_LARGEST[FILTER_NUMBER]:-REMOVE_SMALLEST[FILTER_NUMBER], :]
    print(sorted_events.shape)
    sig_new = [event[1:-1] for event in sorted_events if event[-1] == 1]
    bg_new = [event[1:-1] for event in sorted_events if event[-1] == 0]
    print(len(sig_new[0]))
    print(len(sig_new))
    np.save(OUTPUT_BG_FILE_FORMAT.format(datasets.ROOT, INPUT_ORIGINAL_EXPERIMENT_NAME, FILTER_NUMBER), bg_new)
    np.save(OUTPUT_SIG_FILE_FORMAT.format(datasets.ROOT, INPUT_ORIGINAL_EXPERIMENT_NAME, FILTER_NUMBER), sig_new)


if __name__ == "__main__":
    main()

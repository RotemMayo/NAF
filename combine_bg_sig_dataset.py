import os
import sys
import numpy as np


datasets_path = 'external_maf/datasets/data/lhc'
dataset_name = sys.argv[1]

signal_file_name = os.path.join(datasets_path, "sig_{}.npy".format(dataset_name))
background_file_name = os.path.join(datasets_path, "bg_{}.npy".format(dataset_name))

sig = np.nan_to_num(np.load(signal_file_name))
sig = np.append(sig, np.ones((sig.shape[0], 1)), axis=1)
bg = np.nan_to_num(np.load(background_file_name))
bg = np.append(bg, np.zeros((bg.shape[0], 1)), axis=1)

data = np.append(sig, bg, axis=0)

np.save(os.path.join(datasets_path, "lhc_{}.npy".format(dataset_name)), data)

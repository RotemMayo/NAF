import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def get_mjj(df):
    e1sq = df['pxj1'] ** 2 + df['pyj1'] ** 2 + df['pzj1'] ** 2 + df['mj1'] ** 2
    e2sq = df['pxj2'] ** 2 + df['pyj2'] ** 2 + df['pzj2'] ** 2 + df['mj2'] ** 2
    ptotsq = (df['pxj1'] + df['pxj2']) ** 2 + (df['pyj1'] + df['pyj2']) ** 2 + (df['pzj1'] + df['pzj2']) ** 2
    mjjsq = e1sq + e2sq - ptotsq
    mjjsq[mjjsq < 0] = 0
    return mjjsq ** 0.5


def plot_mj(df, cut1, cut2, bins):
    plt.figure()
    mj = pd.concat([df['mj1'], df['mj2']], axis=1)
    plt.hist2d(x=mj.max(axis=1)[cut1], y=mj.min(axis=1)[cut1], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.figure()
    plt.hist2d(x=mj.max(axis=1)[cut2], y=mj.min(axis=1)[cut2], bins=bins, norm=mpl.colors.LogNorm())
    plt.colorbar()

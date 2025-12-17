# ------------------------------------------------------------------------------
# Store plots of computed results.
# ------------------------------------------------------------------------------

import sys
sys.path.append('..')
from Math import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os
import cv2

def plot_curve (x, y, baselines=None, xaxis="Time (s)", yaxis="Value ()", xlim=None, ylim=None, xsci=True, ysci=True, filename="plot", folder="plots"):
    """
    Arguments:
        x (np.ndarray): x-axis values.
        y (dict {"name": np.ndarray([T])}): Dictionary with entries where the keys are the labels of the plotted lines.
        xaxis (str): Label for the x axis.
        yaxis (str): Label for the y axis.
    """
    os.makedirs(folder, exist_ok=True)

    plt.rcParams.update({'font.size': 7})     # Font size should be max 7pt and min 5pt
    plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
    plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
    mm = 1/25.4
    figsize = (88*mm, 60*mm)
    fig, ax = plt.subplots(figsize=figsize)
    
    if baselines is not None:
        for k in baselines:
            ax.plot(x, baselines[k], '--', label=k)
    for k in y:
        ax.plot(x, y[k], label=k)
        
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xsci:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if ysci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    ax.set_axisbelow(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5, fancybox=True, shadow=False)

    fig.savefig(f"{folder}/{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f"{folder}/{filename}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"\033[96mSaved plot to '{folder}/{filename}.png'\033[0m")













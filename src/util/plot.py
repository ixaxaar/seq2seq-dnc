#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import numpy as np
import seaborn as sns
sns.set_style("darkgrid")


def plot_loss(losses, save=False, name='loss'):
    plt.plot(np.array(losses))
    if save:
        plt.savefig(name + '.jpg', dpi=100)
    return plt


def show_plot(points):
    plt.figure(figsize=(18, 10), dpi=80)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=200)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    return plt

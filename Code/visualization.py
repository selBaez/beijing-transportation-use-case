"""
This module contains the main plotting functions shared across scripts.
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import random

import paths

def _plotDistributionCompare(sample1, sample2, variable_name, labels, bins=None, xticks=None):
    """
    Plot variables distribution with frequency histogram
    """
    # Plot variable frequency histogram
    fig, ax = plt.subplots()

    # Make plot pretty
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Bins
    if bins == 'Auto':
        bins = max(max(sample1), max(sample2)) - min(min(sample1), min(sample2))

    plt.hist([sample1, sample2], label=labels, color=['#578ac1', '#57c194'], bins=bins)

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='center')

    plt.legend(loc='upper right')
    plt.xlabel(variable_name)

    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'.png', format='png')

def _plotDistribution(sample, variable_name, column_name, bins=None, xticks=None):
    """
    Plot variables distribution with frequency histogram
    """
    fig, ax = plt.subplots()

    # Make plot pretty
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Bins
    if bins == 'Auto':
        bins = max(sample[column_name]) - min(sample[column_name])

    # Plot
    sample[column_name].plot.hist(ax=ax, bins=bins, color='#578ac1')

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='center')
    plt.xlabel(variable_name)

    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'.png', format='png')

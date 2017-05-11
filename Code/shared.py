"""
This module contains the main functions shared across scripts.
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns
import random

import paths

def _matrixHeatmap(name, matrix, n_classes, classes):
    """
    Create heat confusion matrix and save to figure
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, vmin=-1, vmax=1)

    plt.xticks(range(n_classes), classes, rotation=70, ha='center', fontsize=8)
    plt.yticks(range(n_classes), reversed(classes), rotation=0, va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'heatmaps/'+name+'.png', format='png')

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

def _plotSeriesCorrelation(sample, variable1, variable2):
    """
    Scatter plot of correlated series in dataframe
    """
    fig, ax = plt.subplots()
    sample.plot(x=variable1,y=variable2, ax=ax, kind='scatter')
    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'scatter/'+variable1+'_'+variable2+'.png', format='png')

def _filter(data, condition, motivation):
    """
    Remove records from data due to motivation according to Boolean condition
    """
    recordsBefore = len(data.index)
    data = condition
    recordsLeft = len(data.index)
    recordsRemoved = recordsBefore - recordsLeft
    print("{} records removed due to {}, {} records left".format(recordsRemoved, motivation, recordsLeft))

    return data

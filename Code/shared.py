"""
This module contains the main functions shared across scripts. Mainly done for visualization or Pandas dataframe manipulation.
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns
import random

import paths

################################################## Visualization ##################################################

def _featureBar(name, values, n_features, features):
    """
    Create bar plot for feature importance
    """
    fig, ax = plt.subplots()
    plt.bar(range(n_features), values)

    plt.xticks(range(n_features), features, rotation=70, ha='center', fontsize=8)
    plt.xlabel(name)
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'bar/'+name+'.png', format='png')
    plt.close()

def _matrixHeatmap(name, matrix, n_attributes, attributes):
    """
    Create heatmap of correlations
    Not annotated values ranging from -1 to 1
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, vmin=-1, vmax=1)

    plt.xticks(range(n_attributes), attributes, rotation=70, ha='center', fontsize=8)
    plt.yticks(range(n_attributes), reversed(attributes), rotation=0, va='bottom', fontsize=8)
    plt.xlabel(name)
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'heatmaps/'+name+'.png', format='png')
    plt.close()

def _classificationHeatmap(name, matrix, n_classes, classes):
    """
    Create heatmap for confusion matrix
    Annotated values ranging from 0 to 1 (as they represent probabilities)
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="f", vmin=0, vmax=1)

    plt.xticks(range(n_classes), classes, rotation=0, ha='left', fontsize=15)
    plt.yticks(range(n_classes), reversed(classes), rotation=0, va='bottom', fontsize=15)
    plt.xlabel(name)
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'heatmaps/'+name+'.png', format='png')
    plt.close()

def _featureSliceHeatmap(name, matrix):
    """
    Create heatmap of slice of user cube representing a feature in their monthly trips
    Not annotated values with range according to distribution
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix)

    plt.xticks(range(1, matrix.shape[1]+1), range(1, matrix.shape[1]+1), rotation=0, ha='right', fontsize=11)
    plt.yticks(range(matrix.shape[0]), reversed(range(matrix.shape[0])), rotation=0, va='bottom', fontsize=11)
    plt.xlabel('Days')
    plt.ylabel('Hours')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'heatmaps/'+name+'.png', format='png')
    plt.close()

def _plotDistributionCompare(sample1, sample2, variable_name, labels, bins=None, xticks=None):
    """
    Plot comparison of variables distribution with frequency histogram
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

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='right', fontsize=8)

    plt.hist([sample1, sample2], label=labels, color=['#578ac1', '#57c194'], bins=bins)

    plt.legend(loc='upper right')
    plt.xlabel(variable_name)
    plt.tight_layout()

    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'.png', format='png')
    plt.close()

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
    # sample[column_name].plot.hist(ax=ax, bins=bins, color='#578ac1')
    sns.distplot(sample[column_name], ax=ax, bins=bins, color='#578ac1')

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='center')
    plt.xlabel(variable_name)

    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'.png', format='png')
    plt.close()

def _plotSeriesCorrelation(sample, variable1, variable2):
    """
    Scatter plot of correlated series in dataframe
    """
    fig, ax = plt.subplots()
    sample.plot(x=variable1,y=variable2, ax=ax, kind='scatter')
    plt.tight_layout()
    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'scatter/'+variable1+'_'+variable2+'.png', format='png')
    plt.close()

def _plotPie(name, sizes, labels):
    """
    Pie chart
    """
    fig, ax = plt.subplots()
    pie = plt.pie(sizes, colors=['#d83f3a', '#cec650', '#287fd6', '#e59c60', '#5fcc6c', '#a888d8'], startangle=120, autopct='%1.2f%%')

    plt.legend(pie[0], labels, loc="lower left")
    plt.axis('equal')
    plt.tight_layout()
    # Save
    plt.savefig(paths.PLOT_DIR_DEFAULT+'pie/'+name+'.png', format='png')
    plt.close()

def _lowDimFeaturesScatter(feature_name, features, labels=None):
    """
    Scatter plot representing the low dimensional features
    """
    fig, ax = plt.subplots()

    if labels == None:
        plt.scatter(features[:,0], features[:,1])
    else:
        plt.scatter(features[:,0], features[:,1], c=labels.astype(int))


    plt.title('Low dimenasional features')
    plt.tight_layout()
    plt.savefig(paths.PLOT_DIR_DEFAULT+'scatter/'+feature_name+'.png', format='png')
    plt.close()


###################################################### Pandas ######################################################

def _filter(data, condition, motivation):
    """
    Remove records from data due to motivation according to Boolean condition
    Return filtered data and number of records removed
    """
    recordsBefore = len(data.index)
    data = condition
    recordsLeft = len(data.index)
    recordsRemoved = recordsBefore - recordsLeft
    print("{} records removed due to {}, {} records left".format(recordsRemoved, motivation, recordsLeft))

    return data, recordsRemoved

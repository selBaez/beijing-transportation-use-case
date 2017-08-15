"""
This module contains the main functions shared across scripts. Mainly done for visualization or Pandas dataframe manipulation.
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random, os

import warnings
warnings.simplefilter("ignore")

import paths

################################################## Visualization ##################################################

def _tsneScatter(feature_name, features, labels=None):
    """
    Scatter plot representing the low dimensional features
    """
    fig = plt.figure()
    plt.clf()

    if labels is None:
        plt.scatter(features[:,0], features[:,1])
    else:
        typesLabel = list(set(labels))
        colors = sns.color_palette("husl", len(typesLabel))

        for typeL in typesLabel:
            condition = labels == typeL
            label = 'Label:' + str(int(typeL)) + ', Num samples: ' + str(len(features[condition,0]))
            plt.scatter(features[condition,0], features[condition,1], color = colors[int(typeL)], label=label)

    plt.legend(numpoints=1, loc="upper left")
    plt.title(feature_name)
    plt.tight_layout()

    directory = paths.PLOT_DIR_DEFAULT+'scatter/TSNE/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+feature_name+'.png', format='png')
    plt.close()

def _stackedFeatureBar(scores, methods, n_features, features, testName, fileName):
    """
    Create stacked bar plot for feature selection
    """
    fig, ax = plt.subplots()
    lastValues = np.zeros(shape=scores[0].shape)

    colors = sns.color_palette("husl", len(methods))

    for i, values in enumerate(scores):
        plt.bar(range(n_features), values, color=colors[i], bottom=lastValues)
        lastValues = lastValues + values

    plt.xticks(range(n_features), features, rotation=70, ha='center', fontsize=8)
    plt.xlabel('Attributes')
    plt.ylabel('Scores')
    plt.legend(methods, loc='upper left')
    plt.title(testName+': '+fileName)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'bar/'+testName+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _featureBar(values, n_features, features, testName, fileName):
    """
    Create bar plot for scoring attributes
    """
    fig, ax = plt.subplots()
    plt.bar(range(n_features), values)

    plt.xticks(range(n_features), features, rotation=70, ha='center', fontsize=8)
    plt.xlabel('Attributes')
    plt.ylabel('Scores')
    plt.title(testName+': '+fileName)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'bar/'+testName+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _correlationHeatmap(matrix, n_attributes, attributes, className):
    """
    Create heatmap of correlations
    Not annotated values ranging from -1 to 1
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, vmin=-1, vmax=1)

    plt.xticks(range(n_attributes), attributes, rotation=70, ha='center', fontsize=8)
    plt.yticks(range(n_attributes), reversed(attributes), rotation=0, va='bottom', fontsize=8)
    plt.title(className+' Correlation')
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'heatmaps/correlations/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+className+'.png', format='png')
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

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'heatmaps/classification/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+name+'.png', format='png')
    plt.close()

def _featureSliceHeatmap(matrix, featureName, className, code):
    """
    Create heatmap of slice of user cube representing a feature in their monthly trips
    Not annotated values with range according to distribution
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, vmin=0, vmax=1, cbar=False)

    plt.xticks(range(1, matrix.shape[1]+1), range(1, matrix.shape[1]+1), rotation=0, ha='right', fontsize=11)
    plt.yticks(range(matrix.shape[0]), reversed(range(matrix.shape[0])), rotation=0, va='bottom', fontsize=11)
    plt.xlabel('Days')
    plt.ylabel('Hours')
    plt.title(featureName+' feature slice for '+className)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'heatmaps/cubes/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = paths.PLOT_DIR_DEFAULT+'heatmaps/cubes/'+className+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = paths.PLOT_DIR_DEFAULT+'heatmaps/cubes/'+className+'/'+code+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+featureName+'.png', format='png')
    plt.close()

def _plotDistributionCompare(sample1, sample2, variable_name, fileName, labels, bins=None, xticks=None):
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

    plt.hist([sample1, sample2], label=labels, color=['#578ac1', '#57c194'], bins=bins)

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='right', fontsize=8)

    plt.legend(loc='upper right')
    plt.xlabel(variable_name)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _plotDistribution(sample, variable_name, condition, fileName, bins=None, xticks=None):
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
        bins = max(sample) - min(sample)

    # Plot
    sns.distplot(sample, ax=ax, bins=bins, color='#578ac1')

    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0), ha='center')

    plt.xlabel(variable_name)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = paths.PLOT_DIR_DEFAULT+'histograms/'+variable_name+'/'+condition+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _plotSeriesCorrelation(sample, variable1, variable2, name, condition, fileName):
    """
    Scatter plot of correlated series in dataframe
    """
    fig, ax = plt.subplots()
    sample.plot(x=variable1,y=variable2, ax=ax, kind='scatter')
    plt.xlabel(name)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'scatter/'+name+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+fileName+'-'+condition+'.png', format='png')
    plt.close()

def _plotPie(sizes, labels, name, fileName):
    """
    Pie chart
    """
    fig, ax = plt.subplots()
    pie = plt.pie(sizes, colors=['#d83f3a', '#cec650', '#287fd6', '#e59c60', '#5fcc6c', '#a888d8'], startangle=120, autopct='%1.2f%%')

    plt.legend(pie[0], labels, loc="lower left")
    plt.axis('equal')
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'pie/'+name+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _sampleCalculateStd(x, y, xlabel, ylabel, title):
    """
    Fill between plot according to mean and standard deviation
    """
    x, y, std = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x, y) if xVal==a]), np.std([yVal for a, yVal in zip(x, y) if xVal==a])) for xVal in set(x)))

    fig, ax = plt.subplots()
    plt.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std), color="#629cdb")
    plt.plot(x, y, color="#3F5D7D", lw=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.set_xticks(x)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'fillBetween/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+title+'.png', format='png')
    plt.close()

def _sampleWithStd(x, y, std, xlabel, ylabel, title):
    """
    Fill between plot according to mean and standard deviation
    """
    fig, ax = plt.subplots()
    plt.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std), color="#629cdb")
    plt.plot(x, y, color="#3F5D7D", lw=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.set_xticks(x)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'fillBetween/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save
    plt.savefig(directory+title+'.png', format='png')
    plt.close()

def _volumeBar(volumeArray, fileName):
    """
    Create bar plot for scoring attributes
    """
    n_days = len(volumeArray)
    days = volumeArray[:,0]
    weekdayClass = volumeArray[:,1]
    volumes = volumeArray[:,2]

    fig, ax = plt.subplots()

    colors = sns.color_palette("husl", 7)
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

    for numWeekday, nameWeekday in weekdays.items():
        condition = weekdayClass == numWeekday
        plt.bar(days[condition]-1, volumes[condition], color=colors[numWeekday], label=nameWeekday)

    plt.xticks(range(n_days), days, rotation=70, ha='center', fontsize=8)
    plt.xlabel('Days')
    plt.ylabel('Volume of records')
    plt.legend(numpoints=1, loc="upper left")
    plt.title(fileName)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'bar/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+fileName+'.png', format='png')
    plt.close()

def _vocabCum(vocabularyFile, fileName):
    """
    Create bar plot for scoring attributes
    """
    n_days = len(vocabularyFile)
    days = vocabularyFile[:,0]
    cummulative = np.cumsum(vocabularyFile[:,1])

    fig, ax = plt.subplots()

    plt.plot(days, cummulative, color='r')

    plt.xticks(range(n_days), days, rotation=70, ha='center', fontsize=8)
    plt.xlabel('Days')
    plt.ylabel('Size of vocabulary')
    plt.title(fileName)
    plt.tight_layout()

    # Deal with folders that do not exist
    directory = paths.PLOT_DIR_DEFAULT+'scatter/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+fileName+'.png', format='png')
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

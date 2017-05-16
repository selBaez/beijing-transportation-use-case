# -*- coding: utf-8 -*-

"""
This module cleans and formats all the Yikatong smart card records.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random, cPickle, csv
from sklearn.preprocessing import StandardScaler

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore 'PATH_LINK', 'START_TIME', 'END_TIME', and 'TRIP_DETAILS'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(4)+range(5,10)+range(12,14)+range(15,32))#, parse_dates=[0,8,9])
    print("{} records loaded".format(len(data.index)))

    return data

def _matchLabel(code, commutersCodes, nonCommutersCodes):
    """
    Match commuters to label 1, non commuters to label 0 and codes without information to NaN
    """
    if code in commutersCodes:
        return 1
    elif code in nonCommutersCodes:
        return 0
    else:
        return None

def _labelData(data, labelsDir):
    """
    Get card codes whose label is available and return commuters and non-commuters datasets
    """
    # Load label sets
    commuters = np.loadtxt(labelsDir+'commuterCardCodes.txt')
    non_commuters = np.loadtxt(labelsDir+'nonCommuterCardCodes.txt')

    # Assign label
    data['LABEL'] = data['CARD_CODE'].apply(lambda x : _matchLabel(x, commuters, non_commuters) )

    # Eliminate records without labels
    data, numUnlabeled = shared._filter(data, data[~data['LABEL'].isnull()], "no label available")

    if FLAGS.plot_distr == 'True':
        shared._plotPie('unlabeled', [numUnlabeled, len(data.index)],\
         ['Unlabeled', 'Labeled'])

    return data

def _visualize(data, text, general=False):
    # Sample 'size' random points
    size = 15 if FLAGS.scriptMode == 'short' else 450

    indices = random.sample(data.index, size)
    sample = data.ix[indices]

    if general == True:
        # Plot general features
        print("Plotting general distributions")
        shared._plotDistributionCompare(sample['START_HOUR'], sample['END_HOUR'], 'Hour of trip', labels=['Start', 'End'], bins=24, xticks=[0, 24])
        shared._plotDistribution(sample, 'Number of trips', 'NUM_TRIPS', bins='Auto', xticks=[0.0, 8.0])
        shared._plotDistributionCompare(sample['ON_AREA'], sample['OFF_AREA'], 'District', labels= ['Boarding', 'Alighting'], bins=18) #Range 1:18
        shared._plotDistributionCompare(sample['ON_TRAFFIC'], sample['OFF_TRAFFIC'], 'Small traffic area', labels= ['Boarding', 'Alighting'], bins=20) #Range 1:1911
        shared._plotDistributionCompare(sample['ON_MIDDLEAREA'], sample['OFF_MIDDLEAREA'], 'Middle traffic area', labels= ['Boarding', 'Alighting'], bins=20) #Range 1:389
        shared._plotDistributionCompare(sample['ON_BIGAREA'], sample['OFF_BIGAREA'], 'Big traffic area', labels= ['Boarding', 'Alighting'], bins=20) #Range 1:60
        shared._plotDistributionCompare(sample['ON_RINGROAD'], sample['OFF_RINGROAD'], 'Ring road', labels= ['Boarding', 'Alighting'], bins=6) #Range 1:6

    # Plot features to be standardized
    print("Plotting "+text+" travel time and distance distributions")
    shared._plotDistribution(sample, 'Travel time '+text, 'TRAVEL_TIME', bins=20)
    shared._plotDistribution(sample, 'Travel distance '+text, 'TRAVEL_DISTANCE', bins=20)
    shared._plotDistribution(sample, 'Total transfer time '+text, 'TRANSFER_TIME_SUM', bins=20)
    shared._plotDistribution(sample, 'Average transfer time '+text, 'TRANSFER_TIME_AVG', bins=20)

    # Plot standardized correlated features: time vs distance
    shared._plotSeriesCorrelation(sample,'TRAVEL_DISTANCE','TRAVEL_TIME')

def _standardize(data):
    """
    Rescale features to have mean 0 and std 1
    """
    # TODO: only fit and transform to train data, and transform test data
    print("Standarize travel time and distance, transfer time total and average")
    scaler = StandardScaler()
    data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']] = scaler.fit_transform(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']])
    # TODO categoical cannot go to one hot, so divide by range?

    return data

def _byCardCode(data):
    """
    structure per card code
    """
    # save groups in new frame with index card code
    data = list(data.groupby('CARD_CODE'))
    print(len(data), ' card codes found')

    # print(list(data[range(4)+range(8,12)].groupby(data['CARD_CODE']))[2])

    return data

def _store(data, className):
    """
    Store data for use in model
    """
    data.to_pickle(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.pkl')
    data.to_csv(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.csv')

def _storeGroups(data, className):
    """
    Store data for use in model
    """
    with open(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.pkl', 'w') as fp: cPickle.dump(data, fp)
    with open(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.csv', 'w') as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerows(data)

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.CLEAN_FILE_DEFAULT+'.csv')

    if FLAGS.plot_distr == 'True':
        print("--------------------- Visualize  general data ---------------------")
        _visualize(data, 'original', general= True)

    print("---------------------- Label and select data ----------------------")
    data = _labelData(data, paths.LABELS_DIR_DEFAULT)

    #print("------------------- Create train  and test sets -------------------")
    #TODO divide and add labels?

    if FLAGS.std == 'True':
        print("-------------------------- Standardizing --------------------------")
        data = _standardize(data)

    if FLAGS.plot_distr == 'True':
        print("----------------- Visualize standardized data -----------------")
        _visualize(data, 'standardized')

    print("-------------------------- Storing  data --------------------------")
    _store(data, 'general')

    print("----------------------- Group by  card code -----------------------")
    data = _byCardCode(data)

    print("-------------------------- Storing  data --------------------------")
    _storeGroups(data, 'groups')

def print_flags():
    """
    Print all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    """
    Main function
    """
    print_flags()
    preprocess()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse route details.')
    parser.add_argument('--plot_distr', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--scriptMode', type = str, default = 'short',
                        help='Run with long  or short dataset.')
    parser.add_argument('--std', type = str, default = 'False',
                        help='Standardize features.')
    #TODO: labeled or unlabeled? (labeled includes searching for codes)

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

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
import random, cPickle
from sklearn.preprocessing import StandardScaler

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 3 'PATH_LINK', 10 and 11 'START_TIME', 'END_TIME', and 13 'TRIP_DETAILS'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(3)+range(4,9)+range(11,13)+range(14,31))#, parse_dates=[0,8,9])
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
    data = shared._filter(data, data[~data['LABEL'].isnull()], "no label available")

    return data

def _standardize(data):
    """
    Rescale features to have mean 0 and std 1
    """

    if FLAGS.plot_distr == 'True':
        # Sample 'size' random points
        size = 100 if FLAGS.scriptMode == 'short' else 2500

        indices = random.sample(data.index, size)
        sample = data.ix[indices]

        # Plot general features
        print("Plotting general distributions")
        shared._plotDistributionCompare(sample['START_HOUR'], sample['END_HOUR'], 'Hour of trip', labels=['Start', 'End'], bins=25, xticks=[0.0, 25.0])
        shared._plotDistribution(sample, 'Number of trips', 'NUM_TRIPS', bins='Auto', xticks=[0.0, 8.0])
        shared._plotDistributionCompare(sample['ON_AREA'], sample['OFF_AREA'], 'District', labels= ['Boarding', 'Alighting'], bins=18, xticks=[1,18])
        shared._plotDistributionCompare(sample['ON_TRAFFIC'], sample['OFF_TRAFFIC'], 'Small traffic area', labels= ['Boarding', 'Alighting'], bins=20, xticks=[1,1911])
        shared._plotDistributionCompare(sample['ON_MIDDLEAREA'], sample['OFF_MIDDLEAREA'], 'Middle traffic area', labels= ['Boarding', 'Alighting'], bins=20, xticks=[1,389])
        shared._plotDistributionCompare(sample['ON_BIGAREA'], sample['OFF_BIGAREA'], 'Big traffic area', labels= ['Boarding', 'Alighting'], bins=20, xticks=[1,60])
        shared._plotDistributionCompare(sample['ON_RINGROAD'], sample['OFF_RINGROAD'], 'Ring road', labels= ['Boarding', 'Alighting'], bins=6, xticks=[1,6])

        # Plot features to be standardized
        print("Plotting original travel time and distance distributions")
        shared._plotDistribution(sample, 'Travel time', 'TRAVEL_TIME', bins=20)
        shared._plotDistribution(sample, 'Travel distance', 'TRAVEL_DISTANCE', bins=20)
        shared._plotDistribution(sample, 'Total transfer time', 'TRANSFER_TIME_SUM', bins=20)
        shared._plotDistribution(sample, 'Average transfer time', 'TRANSFER_TIME_AVG', bins=20)


    # TODO: only fit and transform to train data, and transform test data
    print("Standarize travel time and distance, transfer time total and average")
    scaler = StandardScaler()
    data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']] = scaler.fit_transform(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']])
    # TODO categoical cannot go to one hot, so divide by range?


    if FLAGS.plot_distr == 'True':
        # Use previous sample
        sample = data.ix[indices]

        # Plot standardized features
        print("Plotting standarized travel time and distance distributions")
        shared._plotDistribution(sample, 'Travel time standardized', 'TRAVEL_TIME', bins=20)
        shared._plotDistribution(sample, 'Travel distance standardized', 'TRAVEL_DISTANCE', bins=20)
        shared._plotDistribution(sample, 'Total transfer time standardized', 'TRANSFER_TIME_SUM', bins=20)
        shared._plotDistribution(sample, 'Average transfer time standardized', 'TRANSFER_TIME_AVG', bins=20)

        # Plot standardized correlated features: time vs distance
        shared._plotSeriesCorrelation(sample,'TRAVEL_DISTANCE','TRAVEL_TIME')

    return data

def _store(data, className):
    """
    Store data for use in model
    """
    data.to_pickle(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.pkl')
    data.to_csv(paths.PREPROCESSED_FILE_DEFAULT+'_'+className+'.csv')

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.CLEAN_FILE_DEFAULT+'.csv')

    #print("----------------------- Finding smart codes -----------------------")
    #TODO find records related to given smart codes
    data = _labelData(data, paths.LABELS_DIR_DEFAULT)

    #print("------------------- Create train  and test sets -------------------")
    #TODO divide and add labels?

    print("-------------------------- Standardizing --------------------------")
    # data = _standardize(data)

    print("-------------------------- Storing  data --------------------------")
    _store(data, 'general')

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
    #TODO: labeled or unlabeled? (labeled includes searching for codes)

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

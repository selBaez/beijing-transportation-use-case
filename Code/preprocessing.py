plot# -*- coding: utf-8 -*-

"""
This module preprocess the Yikatong smart card records, labeling the records, standardizing the attributes and creating the 3D representation per user
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import random, cPickle
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
    commutersCodes = np.loadtxt(labelsDir+'commuterCardCodes.txt')
    nonCommutersCodes = np.loadtxt(labelsDir+'nonCommuterCardCodes.txt')

    # Assign label
    data['LABEL'] = data['CARD_CODE'].apply(lambda x : _matchLabel(x, commutersCodes, nonCommutersCodes) )

    # Eliminate records without labels
    data, numUnlabeled = shared._filter(data, data[~data['LABEL'].isnull()], "no label available")

    if FLAGS.plot == 'True':
        shared._plotPie([numUnlabeled, len(data.index)], ['Unlabeled', 'Labeled'], 'unlabeled', FLAGS.file)

    return data

def _visualize(data, condition, general=False):
    # Sample 'size' random points
    size = 15 if FLAGS.scriptMode == 'short' else 450

    indices = random.sample(data.index, size)
    sample = data.ix[indices]

    if general == True:
        # Plot general features
        print("Plotting general distributions")
        shared._plotDistributionCompare(sample['START_HOUR'], sample['END_HOUR'], 'Hour of trip', 'general', FLAGS.file, \
        labels=['Start', 'End'], bins=24, xticks=[0, 24])
        shared._plotDistribution(sample['NUM_TRIPS'], 'Number of trips', 'general', FLAGS.file, bins='Auto', xticks=[0.0, 8.0])
        shared._plotDistributionCompare(sample['ON_AREA'], sample['OFF_AREA'], 'District',  'general', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=18) #Range 1:18
        shared._plotDistributionCompare(sample['ON_TRAFFIC'], sample['OFF_TRAFFIC'], 'Small traffic area',  'general', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:1911
        shared._plotDistributionCompare(sample['ON_MIDDLEAREA'], sample['OFF_MIDDLEAREA'], 'Middle traffic area',  'general', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:389
        shared._plotDistributionCompare(sample['ON_BIGAREA'], sample['OFF_BIGAREA'], 'Big traffic area',  'general', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:60
        shared._plotDistributionCompare(sample['ON_RINGROAD'], sample['OFF_RINGROAD'], 'Ring road',  'general', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=6) #Range 1:6

    # Plot features to be standardized
    print("Plotting "+condition+" travel time and distance distributions")
    shared._plotDistribution(sample['TRAVEL_TIME'], 'Travel time ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRAVEL_DISTANCE'], 'Travel distance ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRANSFER_TIME_SUM'], 'Total transfer time ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRANSFER_TIME_AVG'], 'Average transfer time ', condition, FLAGS.file, bins=20)

    # Plot standardized correlated features: time vs distance
    shared._plotSeriesCorrelation(sample,'TRAVEL_DISTANCE','TRAVEL_TIME', 'Travel distance vs time',condition, FLAGS.file)

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

def _buildCubes(data, createDict):
    """
    Structure per card code
    """
    # Dictionary per class
    if createDict == 'True':
        # Create dictionaries
        print('Creating dictionaries for cubes')
        commutersCubes = {}
        nonCommutersCubes = {}
    else:
        # Load existing dictionaries
        with open(paths.CUBES_DIR_DEFAULT+'commuters.pkl', 'r') as fp: commutersCubes = cPickle.load(fp)
        with open(paths.CUBES_DIR_DEFAULT+'nonCommuters.pkl', 'r') as fp: nonCommutersCubes = cPickle.load(fp)

    # Organize cards per code
    data = list(data.groupby('CARD_CODE'))

    # Sample in case we are debugging
    if FLAGS.scriptMode == 'short':
        data = random.sample(data, 50)

    print(len(data), ' card codes found')


    for userCode, userTrips in data:
        if FLAGS.verbose == 'True': print('code:', userCode, 'label:', userTrips['LABEL'][0], 'number of trips:', userTrips['NUM_TRIPS'][0])

        if userTrips['LABEL'][0] == 1.0:
            # cube is 30 x 24 x 26 TODO: change 16 to 30
            userCube = commutersCubes.setdefault(userCode, np.zeros(shape=(24,16,26)))

            for index, trip in userTrips.iterrows():
                y = trip['START_HOUR']
                x = trip['DAY']
                if FLAGS.verbose == 'True': print('coordinates: ',x, y, 'trip')#, trip[1:-1].values)
                userCube[y, x-1, :] = trip[1:-1].values

            commutersCubes[userCode] = userCube

        else:
            # cube is 30 x 24 x 26
            userCube = nonCommutersCubes.setdefault(userCode, np.zeros(shape=(24,16,26)))

            for index, trip in userTrips.iterrows():
                y = trip['START_HOUR']
                x = trip['DAY']
                if FLAGS.verbose == 'True': print('coordinates: ',x, y, 'trip')#, trip[1:-1].values)
                userCube[y, x-1, :] = trip[1:-1].values

            nonCommutersCubes[userCode] = userCube

    # Sanity check
    code, cube = commutersCubes.popitem()
    if FLAGS.verbose == 'True':
        print('\ncommuter sample', code)
        print(cube[:,:,0])
    if FLAGS.plot == 'True':
        shared._featureSliceHeatmap('commuter-day', cube[:,:,0])

    code, cube = nonCommutersCubes.popitem()
    if FLAGS.verbose == 'True':
        print('\nnon commuter sample', code)
        print(cube[:,:,0])
    if FLAGS.plot == 'True':
        shared._featureSliceHeatmap('NonCommuter-day', cube[:,:,0])

    return commutersCubes, nonCommutersCubes

def _storeDataframe(data, name):
    """
    Store pickle and csv data for use in model
    """
    data.to_pickle(paths.PREPROCESSED_DIR_DEFAULT+FLAGS.file+'_'+name+'.pkl')
    data.to_csv(paths.PREPROCESSED_DIR_DEFAULT+FLAGS.file+'_'+name+'.csv')

def _storeCubes(data, className):
    """
    Store pickle
    """
    with open(paths.CUBES_DIR_DEFAULT+className+'.pkl', 'w') as fp: cPickle.dump(data, fp)

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.CLEAN_DIR_DEFAULT+FLAGS.file+'.csv')

    if FLAGS.plot == 'True':
        print("--------------------- Visualize  general data ---------------------")
        _visualize(data, 'original', general= True)

    print("---------------------- Label and select data ----------------------")
    data = _labelData(data, paths.LABELS_DIR_DEFAULT)

    print("------------------------ Storing dataframe ------------------------")
    _storeDataframe(data, 'labeled')

    if FLAGS.std == 'True':
        print("-------------------------- Standardizing --------------------------")
        data = _standardize(data)
        print("------------------------ Storing dataframe ------------------------")
        _storeDataframe(data, 'labeled-std')

    if FLAGS.plot == 'True':
        print("------------------- Visualize standardized data -------------------")
        _visualize(data, 'standardized')

    print("--------------------------- Build cubes ---------------------------")
    commutersCubes, nonCommutersCubes = _buildCubes(data, FLAGS.create_cubeDict)

    print("-------------------------- Storing cubes --------------------------")
    _storeCubes(commutersCubes, 'commuters')
    _storeCubes(nonCommutersCubes, 'nonCommuters')

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
    parser.add_argument('--file', type = str, default = paths.FILE_DEFAULT,
                        help='File to preprocess')
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse route details.')
    parser.add_argument('--plot', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--scriptMode', type = str, default = 'long',
                        help='Run with long  or short dataset.')
    parser.add_argument('--std', type = str, default = 'True',
                        help='Standardize features.')
    parser.add_argument('--create_cubeDict', type = str, default = 'False',
                        help='Create cube vocabularies from given data. If False, previously saved dictionaries will be loaded')

    #TODO: labeled or unlabeled? (labeled includes searching for codes)

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

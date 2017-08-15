# -*- coding: utf-8 -*-

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
import random, cPickle, os, csv
from sklearn.preprocessing import StandardScaler

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore 'PATH_LINK', 'START_TIME', 'END_TIME', and 'TRANSFER_DETAILS'
    data = pd.read_csv(fileName, index_col='ID', usecols=range(5)+range(6,11)+range(13,15)+range(16,32))
    print("{} records loaded".format(len(data.index)))

    return data

def _matchLabel(code, commutersCodes, nonCommutersCodes, sampleCodes=np.zeros(1)):
    """
    Match commuters to label 1, non commuters to label 0 and codes without information to NaN
    """
    if code in commutersCodes:
        return 1
    elif code in nonCommutersCodes:
        return 0
    elif code in sampleCodes:
        return -1
    else:
        return None

def _labelData(data):
    """
    Get card codes whose label is available and return commuters and non-commuters datasets
    """
    total = len(data.index)

    # Load label sets
    commutersCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'commuterCardCodes.txt')
    nonCommutersCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'nonCommuterCardCodes.txt')

    # Assign label
    data['LABEL'] = data['CARD_CODE'].apply(lambda x : _matchLabel(x, commutersCodes, nonCommutersCodes) )

    # Eliminate records without labels
    data, numUnlabeled = shared._filter(data, data[~data['LABEL'].isnull()], "no label available")

    if FLAGS.plot == 'True':
        shared._plotPie([numUnlabeled, len(data.index)], ['Unlabeled', 'Labeled'], 'unlabeled', FLAGS.file)

    # Save day statistics
    with open(paths.STAT_DIR_DEFAULT+'labeled.txt', 'a') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow([data['DAY'][0], total]+[numUnlabeled, len(data.index)])

    return data

def _reduceData(data):
    """
    Keep card codes whose label is available, and selected sample card codes
    """
    total = len(data.index)

    # Load label sets
    commutersCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'commuterCardCodes.txt')
    nonCommutersCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'nonCommuterCardCodes.txt')
    sampleCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'unknownCardCodes.txt')

    # Assign label
    data['LABEL'] = data['CARD_CODE'].apply(lambda x : _matchLabel(x, commutersCodes, nonCommutersCodes, sampleCodes) )

    # Eliminate records without labels
    data, numRemoved = shared._filter(data, data[~data['LABEL'].isnull()], "card code not selected available")

    if FLAGS.plot == 'True':
        shared._plotPie([numRemoved, len(data.index)], ['Removed', 'Kept'], 'sampled', FLAGS.file)

    # Save day statistics
    with open(paths.STAT_DIR_DEFAULT+'sampled.txt', 'a') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow([data['DAY'][0], total]+[numRemoved, len(data.index)])

    return data

def _visualize(data, condition, general=False):
    # Sample 'size' random points
    size = 5000 if len(data.index) > 5000 else len(data.index)

    indices = random.sample(data.index, size)
    sample = data.ix[indices]

    if general == True:
        # Plot general features
        print("Plotting general distributions")
        shared._plotDistributionCompare(sample['START_HOUR'], sample['END_HOUR'], 'Hour of trip', FLAGS.file, \
        labels=['Start', 'End'], bins=24, xticks=[0, 24])
        shared._plotDistribution(sample['NUM_TRIPS'], 'Number of trips', 'general', FLAGS.file, bins='Auto', xticks=[0.0, 8.0])
        shared._plotDistributionCompare(sample['ON_AREA'], sample['OFF_AREA'], 'District', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=18) #Range 1:18
        shared._plotDistributionCompare(sample['ON_TRAFFIC'], sample['OFF_TRAFFIC'], 'Small traffic area', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:1911
        shared._plotDistributionCompare(sample['ON_MIDDLEAREA'], sample['OFF_MIDDLEAREA'], 'Middle traffic area', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:389
        shared._plotDistributionCompare(sample['ON_BIGAREA'], sample['OFF_BIGAREA'], 'Big traffic area', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=20) #Range 1:60
        shared._plotDistributionCompare(sample['ON_RINGROAD'], sample['OFF_RINGROAD'], 'Ring road', FLAGS.file, \
        labels= ['Boarding', 'Alighting'], bins=6) #Range 1:6

    # Plot features to be standardized
    print("Plotting "+condition+" travel time and distance distributions")
    shared._plotDistribution(sample['TRAVEL_TIME'], 'Travel time ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRAVEL_DISTANCE'], 'Travel distance ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRANSFER_TIME_SUM'], 'Total transfer time ', condition, FLAGS.file, bins=20)
    shared._plotDistribution(sample['TRANSFER_TIME_AVG'], 'Average transfer time ', condition, FLAGS.file, bins=20) # TODO why does this cause error sometimes? singular matrix

    # Plot standardized correlated features: time vs distance
    shared._plotSeriesCorrelation(sample,'TRAVEL_DISTANCE','TRAVEL_TIME', 'Travel distance vs time',condition, FLAGS.file)

def _standardize(data):
    """
    Rescale features to have mean 0 and std 1
    """
    print(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']].mean(axis=0))

    print("Standarize travel time and distance, transfer total and average time")
    scaler = StandardScaler()
    data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']] = scaler.fit_transform(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']])

    return data

def _buildCubes(data, cubeShape=(24,16,26), createDict='False', labeled= 'True'):
    """
    3D structure per card code
    Dictionary has key = card code, value = [cube, label(optional)]
    """
    # Dictionary per class
    if createDict == 'True':
        # Create dictionary
        print('Creating dictionary for cubes')
        userStructures = {}
    else:
        # Load existing corresponding cubes according to label.
        name = 'labeled' if labeled == 'True'  else 'all'
        directory = paths.CUBES_DIR_DEFAULT

        print('Reading from directory:', directory )
        with open(directory+name+'.pkl', 'r') as fp: userStructures = cPickle.load(fp)

    data = list(data.groupby('CARD_CODE'))

    # Sample in case we are debugging
    if FLAGS.scriptMode == 'short':
        data = random.sample(data, 5)

    print(len(data), ' card codes found')

    for userCode, userTrips in data:
        if FLAGS.verbose == 'True': print('code:', userCode, 'label:', userTrips['LABEL'][0], 'number of trips:', userTrips['NUM_TRIPS'][0])

        userCube, userLabel = userStructures.setdefault(userCode, [np.zeros(shape=cubeShape), userTrips['LABEL'][0]])

        for index, trip in userTrips.iterrows():
            y = trip['START_HOUR']
            x = trip['DAY']

            details = trip[1:-1].values

            userCube[int(y), int(x)-1, :] = details

        userStructures[userCode] = [userCube, userLabel]

    for i in range(5):
        code, [cube, label] = random.choice(list(userStructures.items()))
        className = 'Commuter' if label == 1.0 else 'Non-commuter'
        if FLAGS.verbose == 'True': print(className, ' with code: ', str(code))

        if FLAGS.plot == 'True':
            # Plot several feature slices
            print('Plot slices: ', data[0][1].iloc[:, [1,3,4,5,15,16,21,22]].columns.values)

            shared._featureSliceHeatmap(cube[:,:,0], 'Day', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,2], 'Number of trips', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,3], 'Time', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,4], 'Distance', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,14], 'On middle area', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,15], 'Off middle area', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,20], 'On mode', className, str(code))
            shared._featureSliceHeatmap(cube[:,:,21], 'Off mode', className, str(code))

    return userStructures

def _buildVectors(userStructures, labeled= 'True'):
    """
    2D structure per card code
    Dictionary has key = card code, value = [cube, label(optional), flat]
    """
    for code, description in userStructures.items():
        if labeled == 'True':
            [cube, label] = description
            flatCube = cube.flatten(order='F')
            userStructures[code] = [cube, flatCube, label]
        else:
            [cube] = description
            flatCube = cube.flatten(order='F')
            userStructures[code] = [cube, flatCube]

    return userStructures

def _storeDataframe(data, labeled=True):
    """
    Store pickle and csv data for use in model
    """
    labelDirectory = 'labeled/' if labeled  else 'all/'

    # Deal with folders that do not exist
    directory = paths.PREPROCESSED_DIR_DEFAULT+labelDirectory
    if not os.path.exists(directory):
        os.makedirs(directory)

    data.to_csv(directory+FLAGS.file+'.csv')

def _storeStructures(structures, labeled='True'):
    """
    Store pickle
    """
    # Deal with folders that do not exist
    name = 'labeled' if labeled == 'True'  else 'all'
    directory = paths.CUBES_DIR_DEFAULT

    print('Writing to directory:', directory )
    with open(directory+name+'.pkl', 'w') as fp: cPickle.dump(structures, fp)

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.CLEAN_DIR_DEFAULT+FLAGS.file+'.csv')

    if FLAGS.plot == 'True':
        print("--------------------- Visualize  general data ---------------------")
        _visualize(data, 'original', general= True)

    if FLAGS.labeled == 'True':
        print("---------------------- Label and select data ----------------------")
        data = _labelData(data)

        print("-------------------------- Standardizing --------------------------")
        data = _standardize(data)

        print("------------------------ Storing dataframe ------------------------")
        _storeDataframe(data, labeled=True)

    else: # do not select labeled data, save dataframes before standardizing
        print("---------------------- Select data ----------------------")
        data = _reduceData(data)

        print("------------------------ Storing dataframe ------------------------")
        _storeDataframe(data, labeled=False)

        print("-------------------------- Standardizing --------------------------")
        data = _standardize(data)

    if FLAGS.plot == 'True':
        print("------------------- Visualize standardized data -------------------")
        _visualize(data, 'standardized')

    print("--------------------------- Build cubes ---------------------------")
    userStructures = _buildCubes(data, (24,30,26), FLAGS.create_cubeDict, FLAGS.labeled)

    # print("-------------------------- Flatten cubes --------------------------")
    # userStructures = _buildVectors(userStructures, FLAGS.labeled)

    print("----------------------- Storing  structures -----------------------")
    _storeStructures(userStructures, FLAGS.labeled)

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
    parser.add_argument('--verbose', type = str, default = 'True',
                        help='Display parse route details.')
    parser.add_argument('--plot', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--labeled', type = str, default = 'True',
                        help='Choose records which labels are available.')
    parser.add_argument('--scriptMode', type = str, default = 'short',
                        help='Run with long  or short dataset.')
    parser.add_argument('--create_cubeDict', type = str, default = 'True',
                        help='Create cube vocabularies from given data. If False, previously saved dictionaries will be loaded')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

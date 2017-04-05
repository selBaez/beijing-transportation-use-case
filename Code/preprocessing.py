from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############ --- BEGIN default constants --- ############
FILE_NAME_DEFAULT = '../Data/Travel chain sample data(50000).csv'
############ --- END default constants--- ############

def _loadData(fileName):
    """
    Load csv data on numpy
    """
    # Ignore column 2 'DATA_LINK'
    fullData = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[8,9])
    print(len(fullData.index), " records loaded")

    return fullData

def _clean(fullData):
    """
    Remove rows with faulty data
    """
    # Remove rows containing NaN
    noMissing = fullData.dropna()
    missingRecords_num = len(fullData.index) - len(noMissing.index)

    print(missingRecords_num, " records removed due to empty fields, ", len(noMissing.index), " records left")

    # Remove records with travel time <= 0
    timeRelevant = noMissing[noMissing['TRAVEL_TIME'] > 0]
    timeFaultyRecords_num = len(noMissing.index) - len(timeRelevant.index)

    print(timeFaultyRecords_num, " records removed due to travel time <= 0, ", len(timeRelevant.index), " records left")

    # Remove records with travel distance <= 0
    #some still make sense in terms of time and transfers
    distanceRelevant = timeRelevant[timeRelevant['TRAVEL_DISTANCE'] > 0]
    distanceFaultyRecords_num = len(timeRelevant.index) - len(distanceRelevant.index)

    print(distanceFaultyRecords_num, " records removed due to travel distance <= 0, ", len(distanceRelevant.index), " records left")


    #return cleanData

def _parseRoute(data):
    """
    Parse 'TRANSFER_DETAIL' column to get line
    """
    #TODO find way to split by - (representing a transfer)
    data['TRANSFER_DETAIL'] = data['TRANSFER_DETAIL'].str.slice(0,10)
    pass

def _whitening(rawData):
    """
    remove correlations and fit to variance 1
    """
    pass

def _store(preprocessedData):
    """
    Store data for use in model
    """
    pass

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    fullData = _loadData(FLAGS.file_name)

    print("---------------------------- Cleaning -----------------------------")
    cleanData = _clean(fullData)

    print("-------------------------- Parse  route ---------------------------")

    print("------------------------ Extract weekdays -------------------------")

    print("---------------------------- Whitening ----------------------------")

    print("--------------------------- Store  data ---------------------------")

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
    #TODO Make directories if they do not exists yet
    preprocess()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type = str, default = FILE_NAME_DEFAULT,
                        help='Data file to load.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

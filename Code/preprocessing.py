# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############ --- BEGIN default constants --- ############
FILE_NAME_DEFAULT = '../Data/Travel chain sample data(50000).csv'
TRANSLATE_DICT_DEFAULT = {  '轨道' : 'R',
                            '公交' : 'B',
                            '自行车' : 'Z',
                            '线' : 'Line',
                            '号' : '',
                            '夜' : 'Night ',
                            '站' : 'Station ',
                            '小区' : 'District ',
                            '机场' : 'Airport ',
                            '公交场' : 'Bus loop ',
                            '北' : 'North ',
                            '南' : 'South ',
                            '东' : 'East ',
                            '西' : 'West '}
############ --- END default constants--- ############

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 2 'DATA_LINK'
    fullData = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[0,8,9])
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

    # Remove rows with strange transfers
    # TODO

    return distanceRelevant

def _parseRoute(data, chineseDict):
    """
    Parse 'TRANSFER_DETAIL' column to get line
    """
    #TODO find way to split by - (representing a transfer)
    print("Replacing basic Chinese according to dictionary")
    for key in reversed(sorted(chineseDict.keys())):
        data['TRANSFER_DETAIL'] = data['TRANSFER_DETAIL'].str.replace(key,chineseDict[key])

    return data

def _whitening(rawData):
    """
    Remove correlations and fit to variance 1
    """
    print("Standarize travel time")
    print("Standarize travel distance")


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
    withRouteData = _parseRoute(cleanData, TRANSLATE_DICT_DEFAULT)

    print("------------------------ Extract weekdays -------------------------")
    # TODO http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb

    print("---------------------------- Whitening ----------------------------")
    preprocessedData = _whitening(withRouteData)

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

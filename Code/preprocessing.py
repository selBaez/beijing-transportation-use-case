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
    # 7 'TRANSFER_TIME_AVG'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[8,9])

    print(data[:3])

    fig, ax = plt.subplots()
    #data.plot(x='START_TIME', y='TRAVEL_TIME', ax=ax)
    #plt.show()
    return data

def _clean():
    """
    Remove rows with empty fields
    """
    print("records removed due to empty fields")
    print("records removed due to travel distance < 0") #some still make sense in terms of time and transfers
    print("records removed due to travel time = 0")

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
    data = _loadData(FLAGS.file_name)

    print("---------------------------- Cleaning -----------------------------")

    print("-------------------------- Parse  route ---------------------------")

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

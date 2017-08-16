"""
This module samples unlabeled card codes from month data.
"""
from __future__ import print_function

import glob, random
import numpy as np
import pandas as pd
import paths, shared

def _loadFrames(directory):
    """
    Load csv data on pandas
    """
    # Read many sample files
    allFiles = glob.glob(directory + "/*.csv")
    data = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        print("Reading file:", file_)
        df = pd.read_csv(file_, usecols=['CARD_CODE'], header=0)
        list_.append(df)
    data = pd.concat(list_)

    print('\n')
    print(len(data.index), "records loaded")

    return data

def _removeLabeled(data):
    """
    Search for unlabeled codes
    """
    # Load codes sets
    commCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'commuterCardCodes.txt')
    nonComCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'nonCommuterCardCodes.txt')
    labeledCodes = np.concatenate((commCodes, nonComCodes))

    # Eliminate codes that are not in sample codes
    data, _ = shared._filter(data, data[~data['CARD_CODE'].isin(labeledCodes)], "not desired card codes")

    return data

def main(_):
    """
    Main function
    """
    # Load data
    data = _loadFrames(paths.RAW_DIR_DEFAULT)

    # Keep unique
    data.drop_duplicates(['CARD_CODE'], inplace=True)
    print(len(data.index), "unique card codes loaded\n")

    # Filter out labeled
    data = _removeLabeled(data)

    # Sample 'size' random points
    size = 1500 if (len(data.index) > 1500) else len(data.index)
    print('\nChoose ', size, ' codes')

    sample = data.sample(size)
    print(len(sample.index), "unknown card codes sampled\n")

    # Save
    np.savetxt(paths.LABELS_DIR_DEFAULT+'unknownCardCodes.txt', sample.values, '%5.0f')


if __name__ == '__main__':
    main(None)

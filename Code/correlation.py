"""
This module implements training and evaluation of an ensemble model for classification.
Argument parser and general sructure partly based on Deep Learning practicals from UvA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    data = pd.read_csv(fileName, index_col='ID')
    print(len(data.index), " records loaded")

    return data

def correlationAnalysis():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    print(paths.PREPROCESSED_FILE_DEFAULT+'_general.csv')
    data = _loadData(paths.PREPROCESSED_FILE_DEFAULT+'_general.csv')
    # print(data.dtypes)

    print("-------------------------- Correlation --------------------------")
    correlationsMatrix = data.corr()
    # print(correlationsMatrix)

    print("--------------------- Plot correlation matrix ---------------------")
    classes = data.select_dtypes(include=[np.number]).columns.values.tolist()
    n_classes = len(classes)
    shared._matrixHeatmap('correlation', correlationsMatrix, n_classes, classes)

    print("----------------------- Per class  analysis -----------------------")
    # Filter non relevant records
    commutersData = data[data['LABEL'] == 1]
    non_commutersData = data[data['LABEL'] == 0]

    commuters_correlationsMatrix = commutersData.corr()
    non_commuters_correlationsMatrix = non_commutersData.corr()

    shared._matrixHeatmap('commuterCorrelation', commuters_correlationsMatrix, n_classes, classes)
    shared._matrixHeatmap('nonCommuterCorrelation', non_commuters_correlationsMatrix, n_classes, classes)

    print("---------------------------- Predict ------------------------------")


    print("---------------------------- Evaluate -----------------------------")


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
    correlationAnalysis()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

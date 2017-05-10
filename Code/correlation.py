"""
This module implements training and evaluation of an ensemble model for classification.
Argument parser and general sructure partly based on Deep Learning practicals from UvA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

############ --- BEGIN default constants --- ############

############ --- END default constants--- ############

############ --- BEGIN default directories --- ############
LOAD_FILE_DEFAULT = '../Data/sets/preprocessed sample data(50000)-noStand.csv'
PLOT_DIR_DEFAULT = './Plots/'
# SAVE_TO_FILE_DEFAULT = '../Data/sets/preprocessed sample data(50000)'
############ --- END default directories--- ############

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 2 'DATA_LINK'
    data = pd.read_csv(fileName, index_col='ID')#, usecols= range(2)+range(3,23), parse_dates=[0,8,9])
    print(len(data.index), " records loaded")

    return data

def _matrixHeat(name, matrix, n_classes, classes):
    """
    Create heat confusion matrix and save to figure
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix)

    plt.xticks(range(n_classes), classes, rotation=70, ha='center', fontsize=8)
    plt.yticks(range(n_classes), reversed(classes), rotation=0, va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(FLAGS.plot_dir+name+'_Heatmap.png', format='png')

def correlationAnalysis():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(FLAGS.load_file)
    # print(data.dtypes)

    print("-------------------------- Correlation --------------------------")
    correlationsMatrix = data.corr()
    print(correlationsMatrix)

    print("--------------------- Plot correlation matrix ---------------------")
    classes = data.select_dtypes(include=[np.number]).columns.values.tolist()
    n_classes = len(classes)
    _matrixHeat('correlation', correlationsMatrix, n_classes, classes)

    print("---------------------- Forward pass  modules ----------------------")


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
    parser.add_argument('--load_file', type = str, default = LOAD_FILE_DEFAULT,
                        help='Data file to load.')
    parser.add_argument('--plot_dir', type = str, default = PLOT_DIR_DEFAULT,
                        help='Directory to which save plots.')


    FLAGS, unparsed = parser.parse_known_args()
    main(None)

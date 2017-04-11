"""
This module implements training and evaluation of an ensemble model for classification.
Argument parser and general sructure partly based on Deep Learning practicals from UvA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

############ --- BEGIN default constants --- ############
FILE_NAME_DEFAULT = './Previous work/commuting-classifier/dataset.mat'
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
NUM_TREES_DEFAULT = 100
DEPTH_TREES_DEFAULT = 10
############ --- END default constants--- ############

def loadData(fileName):
    """
    Load preprocessed data
    """
    data = sio.loadmat(FLAGS.file_name)

    return data['train_data'], data['train_labels'], data['test_data'], data['test_labels']

def _evaluate():
    """
    Confusion matrix and hinge loss
    """

def train():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    train_data, train_labels, test_data, test_labels = loadData(FLAGS.file_name)

    #print("--------------------------- Create sets ---------------------------")

    print("--------------------------- Build model ---------------------------")
    # Random Forest
    model_rf = ensemble.RandomForestClassifier(n_estimators=FLAGS.num_trees, max_depth=FLAGS.depth_trees)

    print("---------------------- Forward pass  modules ----------------------")
    model_rf.fit(train_data, train_labels.ravel())

    #print("------------------------ Assemble  answers ------------------------")

    print("---------------------------- Predict ------------------------------")
    predictions_rf = model_rf.predict_proba(test_data)[:,1]

    print("---------------------------- Evaluate -----------------------------")
    accuracy_rf = average_precision_score(test_labels, predictions_rf)
    print("accuracy : ", accuracy_rf)

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
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type = str, default = FILE_NAME_DEFAULT,
                        help='Data file to load.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--num_trees', type = int, default = NUM_TREES_DEFAULT,
                        help='Number of trees in random forest.')
    parser.add_argument('--depth_trees', type = int, default = DEPTH_TREES_DEFAULT,
                        help='Depth of trees in random forest.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

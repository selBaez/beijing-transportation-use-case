"""
This module implements training and evaluation of an ensemble model for classification.
Argument parser and general sructure partly based on Deep Learning practicals from UvA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

############ --- BEGIN default constants --- ############
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
############ --- END default constants--- ############

def _evaluate():
    pass

def train():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")

    print("--------------------------- Build model ---------------------------")

    print("---------------------- Forward pass  modules ----------------------")

    print("------------------------ Assemble  answers ------------------------")

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
    #train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

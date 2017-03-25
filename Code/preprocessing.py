from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

def _clean():
    pass

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("--------------------------- Parsing CSV ---------------------------")

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
    #preprocess()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

"""
This module builds a CNN and runs the user cubes through it to produce feature maps for clustering.
"""
import argparse
import numpy as np
import random, cPickle
from sklearn.manifold import TSNE

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

import paths, shared

def _loadData():
    """
    Load user cubes stored as pickle
    """
    with open(paths.CUBES_DIR_DEFAULT+'commuters.pkl', 'r') as fp: commutersCubes = cPickle.load(fp)
    with open(paths.CUBES_DIR_DEFAULT+'nonCommuters.pkl', 'r') as fp: nonCommutersCubes = cPickle.load(fp)

    return commutersCubes, nonCommutersCubes

def _buildMode():
    """
    Build a convolutional autoencoder
    """
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # loss function may be user defined function
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder

def _train(autoencoder, data):
    """
    Train the autoencoder
    """
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir=paths.LOG_DIR_DEFAULT)])

    return autoencoder

def _encode(encoder, data):
    """
    Create low dimensionality representation
    """
    encoded_samples = encoder.predict(data)

    return encoded_samples

def _visualize(encodedData):
    """
    Visualize data with TSNE
    """
    manifold = TSNE(n_components=2, random_state=0)
    features = manifold.fit_transform(encodedData)
    shared._lowDimFeaturesScatter('Encoded', features)

def _store(encodedData):
    """
    Store pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'.pkl', 'w') as fp: cPickle.dump(encodedData, fp)

def featureEngineering():
    print("---------------------------- Load data ----------------------------")
    data = _loadData() # standardized, no label cubes

    print("--------------------------- Build model ---------------------------")
    autoencoder, encoder = _buildModel()

    print("--------------------------- Train model ---------------------------")
    autoencoder = _train(autoencoder, data)

    print("------------------------- Evaluate  model -------------------------")
    encoded_samples = _encode(encoder, data)  # Test data ideally

    print("----------------------- Visualize  features -----------------------")
    _visualize(encoded_samples)

    print("-------------------------- Save features --------------------------")
    _store(encoded_samples)


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
    featureEngineering()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse route details.')
    parser.add_argument('--plot_distr', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--scriptMode', type = str, default = 'long',
                        help='Run with long  or short dataset.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

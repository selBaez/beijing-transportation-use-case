"""
This module builds a CNN and runs the user cubes through it to produce feature maps for clustering.
"""
import argparse
import numpy as np
import random, cPickle

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

import paths, shared

def _loadData():
    """
    Load user cubes stored as pickle
    """
    with open(paths.CUBES_DIR_DEFAULT+'labeled.pkl', 'r') as fp: userStructures = cPickle.load(fp)

    # Array contains three columns: code, vector, label %TODO update to unsupervised cube format
    codes = []
    original = []
    labels = []

    for code, [cube, label] in userStructures.items():
        # Select full weeks only
        # cube = cube[:,:28,:]
        # Format
        codes.append(code)
        original.append(cube)
        labels.append(label)

    return np.asarray(codes), np.asarray(original), np.asarray(labels)

def _buildModel():
    """
    Build a convolutional autoencoder
    """
    input_img = Input(shape=(24, 30, 26))  # adapt this if using `channels_first` image data format
    print(input_img)

    x = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(24,30,26))(input_img)
    print(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(x)

    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # print(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # print(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    print(x)
    encoded = MaxPooling2D((3, 3), padding='same')(x)
    print('encoded:', encoded)

    print('\n\n')
    # at this point the representation is (4, 5, 8) i.e. 96-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    print(x)
    x = UpSampling2D((3, 3))(x)
    print(x)

    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # print(x)
    # x = UpSampling2D((2, 2))(x)
    # print(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    print(x)
    x = UpSampling2D((2, 2))(x)
    print(x)

    decoded = Conv2D(26, (3, 3), activation='sigmoid', padding='same', input_shape=(24,30,26))(x)
    print(decoded)


    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # loss function may be user defined function
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder

def _train(autoencoder, x_train, x_test):
    """
    Train the autoencoder
    """
    autoencoder.fit(x_train, x_train,
                    epochs=10,
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
    print(encodedData.shape)

    encodedData = encodedData.reshape((len(encodedData), np.prod(encodedData.shape[1:])))
    print(encodedData.shape)

    features = manifold.fit_transform(encodedData)
    shared._lowDimFeaturesScatter('Encoded', features)

def _store(encodedData):
    """
    Store pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'.pkl', 'w') as fp: cPickle.dump(encodedData, fp)

def featureEngineering():
    print("---------------------------- Load data ----------------------------")
    [codes, cubes, labels] = _loadData()

    print("--------------------------- Create sets ---------------------------")
    train_data, test_data, train_labels, test_labels = train_test_split(cubes, labels, test_size=0.4)

    print("--------------------------- Build model ---------------------------")
    autoencoder, encoder = _buildModel()

    print("--------------------------- Train model ---------------------------")
    autoencoder = _train(autoencoder, train_data, test_data)

    print("------------------------- Evaluate  model -------------------------")
    encoded_samples = _encode(encoder, cubes)  # Test data ideally

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

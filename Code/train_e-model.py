"""
This module implements training and evaluation of an ensemble model for classification.
Argument parser and general sructure partly based on Deep Learning practicals from UvA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import cross_validation, svm, ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import average_precision_score, confusion_matrix

############ --- BEGIN default constants --- ############
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
NUM_TREES_DEFAULT = 100
DEPTH_TREES_DEFAULT = 10
############ --- END default constants--- ############

############ --- BEGIN default directories --- ############
LOAD_FILE_DEFAULT = './Previous work/commuting-classifier/dataset.mat'
PLOT_DIR_DEFAULT = './Plots/'
# SAVE_TO_FILE_DEFAULT = '../Data/sets/preprocessed sample data(50000)'
############ --- END default directories--- ############

def loadData(fileName):
    """
    Load preprocessed data
    """
    data = sio.loadmat(fileName)

    return data['train_data'], data['train_labels'], data['test_data'], data['test_labels']

def _predict(name, predict_fn, test_data):
    """
    Predict class using model's prefictive function
    Return classes as ordinal, not one hot
    """
    predictions = predict_fn(test_data)

    if len(predictions.shape) > 1:
        print('Need to turn ', name,' predictions into categorical')
        predictions = np.argmax(predictions, axis=1)

    return predictions

def _confusion_matrix(name, true, predicted, n_classes):
    """
    Create heat confusion matrix and save to figure
    """

    # print(true.shape, predicted.shape)
    # print(type(true[0]), type(predicted[0]))
    # print(true[:5])
    # print(predicted[:5])

    if n_classes == 2:
        classes = ['Commuters','Non-Commuters']
        labels = [0, 1]

    confusion_array = confusion_matrix(true, predicted, labels=labels)

    norm_conf = []
    for i in confusion_array:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            value = 0. if float(a) == 0 else (float(j)/float(a))
            tmp_arr.append(value)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

    for x in xrange(n_classes):
        for y in xrange(n_classes):
            ax.annotate(str(confusion_array[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks([], [])
    plt.yticks(range(n_classes), classes)
    plt.savefig(FLAGS.plot_dir+name+'_confusionMatrix.png', format='png')

def _evaluate(name, labels, predictions):
    """
    Confusion matrix and hinge loss
    """
    accuracy = average_precision_score(labels, predictions)
    print(name, " accuracy : ", accuracy)

    _confusion_matrix(name, labels[:,0], predictions, 2)

def train():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    train_data, train_labels, test_data, test_labels = loadData(FLAGS.load_file)

    print("-------------------------- Rename labels --------------------------")
    train_labels[train_labels == -1] = 0
    test_labels[test_labels == -1] = 0

    #print("--------------------------- Create sets ---------------------------")

    print("--------------------------- Build model ---------------------------")
    #################### Individual classifiers ####################
    # Original SVM
    model_svm =  svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, degree=1,gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False)

    # Gaussian

    # Bayes

    #################### Ensemble from the box ####################
    # Random Forest
    # TODO, test bootstrap = True
    model_rf = ensemble.RandomForestClassifier(n_estimators=FLAGS.num_trees, max_depth=FLAGS.depth_trees)

    # AdaBoost with trees
    model_adb = ensemble.AdaBoostClassifier(n_estimators=FLAGS.num_trees)

    # Bagging trees 
    model_bag = ensemble.BaggingClassifier(n_estimators=FLAGS.num_trees)



    # Ensemble perceptrons?




    print("---------------------- Forward pass  modules ----------------------")
    model_svm.fit(train_data, train_labels.ravel())
    model_rf.fit(train_data, train_labels.ravel())
    model_adb.fit(train_data, train_labels.ravel())

    print("---------------------------- Predict ------------------------------")
    predictions_svm = _predict('SVM', model_svm.predict, test_data)
    predictions_rf = _predict('Random forest', model_rf.predict, test_data)
    predictions_adb = _predict('AdaBoost of decision trees', model_adb.predict, test_data)

    print("---------------------------- Evaluate -----------------------------")
    # TODO: cross validation
    _evaluate('SVM', test_labels, predictions_svm)
    _evaluate('Random forest', test_labels, predictions_rf)
    _evaluate('AdaBoost of decision trees', test_labels, predictions_adb)

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
    parser.add_argument('--load_file', type = str, default = LOAD_FILE_DEFAULT,
                        help='Data file to load.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--num_trees', type = int, default = NUM_TREES_DEFAULT,
                        help='Number of trees in random forest.')
    parser.add_argument('--depth_trees', type = int, default = DEPTH_TREES_DEFAULT,
                        help='Depth of trees in random forest.')
    parser.add_argument('--plot_dir', type = str, default = PLOT_DIR_DEFAULT,
                        help='Directory to which save plots.')


    FLAGS, unparsed = parser.parse_known_args()
    main(None)

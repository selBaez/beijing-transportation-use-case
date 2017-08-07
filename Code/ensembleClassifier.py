"""
This module implements training and evaluation of an ensemble model for commuter classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore")
import cPickle
from sklearn import svm, ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

import paths, shared

############ --- BEGIN default constants --- ############
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
NUM_TREES_DEFAULT = 100
DEPTH_TREES_DEFAULT = 10
NUM_CLASSES = 2
CLASSES = ['Non-Commuters', 'Commuters']
LABELS = [0, 1]
############ --- END default constants--- ############

def _loadData():
    """
    Load preprocessed data
    """
    with open(paths.LOWDIM_DIR_DEFAULT+"supervised.pkl", 'r') as fp: data = cPickle.load(fp)

    return data

def _predict(name, predict_fn, test_data):
    """
    Predict class using model's prefictive function
    Return classes as ordinal, not one hot
    """
    predictions = predict_fn(test_data)

    return predictions

def _confusion_matrix(name, true, predicted, n_classes, classes, labels):
    """
    Create heat confusion matrix
    """
    # Calculate confusion matrix
    confusion_array = confusion_matrix(true, predicted, labels=labels)

    # Class accuracy
    norm_conf = []
    for i in confusion_array:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            value = 0. if float(a) == 0 else (float(j)/float(a))
            tmp_arr.append(value)
        norm_conf.append(tmp_arr)

    # Plot
    shared._classificationHeatmap(name, np.array(norm_conf), n_classes, classes)

def _evaluate(name, labels, predictions):
    """
    Confusion matrix and accuracy
    """
    accuracy = average_precision_score(labels, predictions)
    print(name)
    print("One split accuracy : %0.2f" % (accuracy))

    # _confusion_matrix(name, labels, predictions, n_classes=NUM_CLASSES, classes=CLASSES, labels=LABELS)

def _crossVal(model, features, labels, cv=5):
    """
    Evaluate using cross validation
    """
    scores = cross_val_score(model, features, labels, cv=5)
    print("Cross validation accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std() * 2))

    return scores.mean()

def train():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    [codes, features, labels] = _loadData()

    print("--------------------------- Create sets ---------------------------")
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.4)

    print("--------------------------- Build model ---------------------------")
    #################### Individual classifiers ####################
    # Original SVM
    model_svm =  svm.SVC(C=0.1, degree=1, kernel='linear', tol=0.001)

    # Gaussian Process
    model_gp = GaussianProcessClassifier()

    # Naive Bayes
    model_gnb = GaussianNB()

    # Perceptron
    model_pct = MLPClassifier()

    #################### Ensemble from the box ####################
    # Random Forest
    model_rf = ensemble.RandomForestClassifier(n_estimators=FLAGS.num_trees, max_depth=FLAGS.depth_trees)

    # AdaBoost with trees
    model_adb = ensemble.AdaBoostClassifier(n_estimators=FLAGS.num_trees)

    # Bagging trees
    model_bag = ensemble.BaggingClassifier(n_estimators=FLAGS.num_trees)

    print("---------------------- Forward pass  modules ----------------------")
    model_svm.fit(train_data, train_labels.ravel())
    model_gp.fit(train_data, train_labels.ravel())
    model_gnb.fit(train_data, train_labels.ravel())
    model_pct.fit(train_data, train_labels.ravel())
    model_rf.fit(train_data, train_labels.ravel())
    model_adb.fit(train_data, train_labels.ravel())
    model_bag.fit(train_data, train_labels.ravel())

    print("---------------------------- Predict ------------------------------")
    predictions_svm = _predict('SVM', model_svm.predict, test_data)
    predictions_gp = _predict('Gaussian Process', model_gp.predict, test_data)
    predictions_gnb = _predict('Gaussian Naive Bayes', model_gnb.predict, test_data)
    predictions_pct = _predict('Perceptron', model_pct.predict, test_data)
    predictions_rf = _predict('Random forest', model_rf.predict, test_data)
    predictions_adb = _predict('AdaBoost of decision trees', model_adb.predict, test_data)
    predictions_bag = _predict('Bagging of decision trees', model_bag.predict, test_data)

    print("---------------------------- Evaluate -----------------------------")
    # TODO refactor
    scores = []

    _evaluate('SVM', test_labels, predictions_svm)
    avScore = _crossVal(model_svm, features, labels)
    scores.append(['SVM', model_svm, avScore])

    _evaluate('Gaussian Process', test_labels, predictions_gp)
    avScore = _crossVal(model_gp, features, labels)
    scores.append(['Gaussian Process', model_gp, avScore])

    _evaluate('Gaussian Naive Bayes', test_labels, predictions_gnb)
    avScore = _crossVal(model_gnb, features, labels)
    scores.append(['Gaussian Naive Bayes', model_gnb, avScore])

    _evaluate('Perceptron', test_labels, predictions_pct)
    avScore = _crossVal(model_pct, features, labels)
    scores.append(['Perceptron', model_pct, avScore])

    _evaluate('Random forest', test_labels, predictions_rf)
    avScore = _crossVal(model_rf, features, labels)
    scores.append(['Random forest', model_rf, avScore])

    _evaluate('AdaBoost of decision trees', test_labels, predictions_adb)
    avScore = _crossVal(model_adb, features, labels)
    scores.append(['AdaBoost of decision trees', model_adb, avScore])

    _evaluate('Bagging of decision trees', test_labels, predictions_bag)
    avScore = _crossVal(model_bag, features, labels)
    scores.append(['Bagging of decision trees', model_bag, avScore])

    print("-------------------------- Select models --------------------------")
    # TODO histogram plot
    scores = np.array(scores)

    k = 3
    selectedModelIdx = np.argsort(scores[:,2])[-k:]
    print("Best {}: {}".format(k, scores[:,0][selectedModelIdx]))

    print("---------------------------- Ensemble ----------------------------")
    estimators = scores[selectedModelIdx,:2]
    print(estimators.shape)

    model = VotingClassifier(estimators)
    avScore = _crossVal(model, features, labels)

    # TODO Save model

    print("----------------------------- Predict -----------------------------")
    model.fit(features, labels)
    predictions = model.predict(features)

    # TODO Save card codes and labels


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

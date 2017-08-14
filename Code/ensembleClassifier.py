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
    Accuracy on test set and confusion matrix
    """
    accuracy = average_precision_score(labels, predictions)
    print(name)
    print("One split accuracy : %0.2f" % (accuracy))

    if FLAGS.plot == 'True':
        _confusion_matrix(name, labels, predictions, n_classes=NUM_CLASSES, classes=CLASSES, labels=LABELS)

def _crossVal(model, features, labels, cv=5):
    """
    Evaluate using cross validation
    """
    scores = cross_val_score(model, features, labels, cv=5)
    print("Cross validation accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))

    return scores.mean(), scores.std()

def _individualClassifier(name, model):
    """
    Train and test an individual classifier
    """
    model.fit(train_data, train_labels.ravel())
    predictions = model.predict(test_data)

    _evaluate(name, test_labels, predictions)
    avScore, stdScore = _crossVal(model, features, labels)

    return [name, model, avScore]

def _ensembleClassifiers(individualScores):
    """
    Build ensemble models gradually increasing the number of classifiers included
    """
    ensembleScores = []

    for k in range(2, len(individualScores)+1):
        # Select best classifiers
        selectedModelIdx = np.argsort(individualScores[:,2])[-k:]
        estimators = individualScores[selectedModelIdx,:2]
        print("Best {}: {}".format(k, individualScores[:,0][selectedModelIdx]))

        # Ensemble and evaluate
        model = VotingClassifier(estimators, voting='soft')
        avScore, stdScore = _crossVal(model, features, labels)
        ensembleScores.append([model, avScore])

    ensembleScores = np.array(ensembleScores)

    return ensembleScores

def _save(model, commuters, nonCommuters):
    """
    Save trained model
    """
    directory = paths.MODELS_DIR_DEFAULT
    with open(directory+"ensembleClassifier.pkl", "w") as fp: cPickle.dump(model, fp)

    np.savetxt(paths.LABELS_DIR_DEFAULT+'classifiedCommuters.txt', commuters, '%5.0f')
    np.savetxt(paths.LABELS_DIR_DEFAULT+'classifiedNonCommuters.txt', nonCommuters, '%5.0f')

def train():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    global features, labels
    [codes, features, labels] = _loadData()
    print(len(codes), " records loaded")

    print("--------------------------- Create sets ---------------------------")
    global train_data, test_data, train_labels, test_labels
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.3)

    print("--------------------- Build weak  classifiers ---------------------")
    #################### Individual classifiers ####################
    # Original SVM
    model_svm =  svm.SVC(C=0.1, degree=1, kernel='linear', tol=0.0001, probability=True)

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

    print("---------------------------- Evaluate -----------------------------")
    individualScores = []

    individualScores.append(_individualClassifier('SVM', model_svm))
    individualScores.append(_individualClassifier('Gaussian Process', model_gp))
    individualScores.append(_individualClassifier('Gaussian Naive Bayes', model_gnb))
    individualScores.append(_individualClassifier('Perceptron', model_pct))
    individualScores.append(_individualClassifier('Random forest', model_rf))
    individualScores.append(_individualClassifier('AdaBoost of decision trees', model_adb))
    individualScores.append(_individualClassifier('Bagging of decision trees', model_bag))

    individualScores = np.array(individualScores)

    print("---------------------- Build ensemble models ----------------------")
    ensembleScores = _ensembleClassifiers(individualScores)

    print("------------------------ Select best model ------------------------")
    selectedModelIdx = np.argsort(ensembleScores[:,1])[-1:]
    model = ensembleScores[selectedModelIdx,0][0]
    print("Final model contains the top {} weak classifiers".format(selectedModelIdx+2))

    print("----------------------------- Predict -----------------------------")
    model.fit(features, labels)
    predictions = model.predict(features) # TODO predict on all

    print("------------------------ Match card  codes ------------------------")
    commuters = codes[predictions == 1]
    nonCommuters = codes[predictions == 0]

    print("------------------------------  Save ------------------------------")
    _save(model, commuters, nonCommuters)

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
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--num_trees', type = int, default = NUM_TREES_DEFAULT,
                        help='Number of trees in random forest.')
    parser.add_argument('--depth_trees', type = int, default = DEPTH_TREES_DEFAULT,
                        help='Depth of trees in random forest.')


    FLAGS, unparsed = parser.parse_known_args()
    main(None)

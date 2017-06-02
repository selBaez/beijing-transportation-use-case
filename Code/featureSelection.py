"""
This module implements feature selection for commuters classification via correlation tests
"""
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    data = pd.read_csv(fileName, index_col='ID')
    print("{} records loaded").format(len(data.index))

    return data

def _correlationAnalysis(data, attributes, n_attributes):
    """
    Plot correlations according to type
    """
    commutersData = data[data['LABEL'] == 1]
    non_commutersData = data[data['LABEL'] == 0]

    correlationsMatrix = data.corr()
    commuters_correlationsMatrix = commutersData.corr()
    non_commuters_correlationsMatrix = non_commutersData.corr()

    scores = np.absolute(correlationsMatrix.values[1:-1,-1])

    if FLAGS.plot == 'True':
        print("               ----- Plot correlation heatmap -----                ")
        shared._correlationHeatmap(correlationsMatrix, n_attributes, attributes, 'General')
        shared._correlationHeatmap(commuters_correlationsMatrix, n_attributes, attributes, 'Commuter')
        shared._correlationHeatmap(non_commuters_correlationsMatrix, n_attributes, attributes, 'Non Commuter')

        shared._featureBar(scores, n_attributes-2, attributes[1:-1], 'Correlation', 'to label')

    # TODO: sets of attributes that overlap a lot?

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _featureImportance(samples, labels, attributes, n_attributes):
    """
    Measure feature importance by using a Tree
    """
    model = ExtraTreesClassifier()
    model.fit(samples, labels)

    if FLAGS.plot == 'True':
        shared._featureBar(model.feature_importances_, n_attributes-2, attributes[1:-1], 'Feature Importance', 'scores')

    scores = model.feature_importances_

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _chi2(samples, labels, attributes, n_attributes):
    """
    Calculate chi squared scores
    """
    chi2val, pval = chi2(samples, labels)

    if FLAGS.plot == 'True':
        shared._featureBar(chi2val, n_attributes-2, attributes[1:-1], 'Chi squared', 'scores')
        shared._featureBar(pval, n_attributes-2, attributes[1:-1], 'Chi squared', 'p values')

    scores = chi2val

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _anova(samples, labels, attributes, n_attributes):
    """
    Calculate F values via ANOVA
    """
    fval, pval2 = f_classif(samples, labels)

    if FLAGS.plot == 'True':
        shared._featureBar(fval, n_attributes-2, attributes[1:-1], 'F values', 'scores')
        shared._featureBar(pval2, n_attributes-2, attributes[1:-1], 'F values', 'p values')

    scores = fval

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def selectFeatures():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.PREPROCESSED_DIR_DEFAULT+'labeled/original/'+FLAGS.file+'.csv')

    print("----------------------- Extract  attributes -----------------------")
    attributes = data.select_dtypes(include=[np.number]).columns.values.tolist()
    n_attributes = len(attributes)

    # Exclude card code
    samples = data[attributes[1:-1]].values
    labels = data[attributes[-1]].values

    print("---------------------- Analyze  correlations ----------------------")
    scores_cr = _correlationAnalysis(data, attributes, n_attributes)

    print("---------------------- Feature  importance ------------------------")
    scores_fi = _featureImportance(samples, labels, attributes, n_attributes)

    print("------------------------ Chi squared  test ------------------------")
    scores_c2 = _chi2(samples, labels, attributes, n_attributes)

    print("----------------------------- ANOVA-F -----------------------------")
    scores_fv = _anova(samples, labels, attributes, n_attributes)

    print("------------------------ Aggregate  scores ------------------------")
    scores = [scores_cr, scores_fi, scores_c2, scores_fv]
    methods = ['correlation', 'importance', 'chi2', 'f-value']

    shared._stackedFeatureBar(scores, methods, n_attributes-2, attributes[1:-1], 'Combined', 'scores')


    # cv = KFold(2)
    # selection = SelectKBest(5)
    # Feature Union


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
    selectFeatures()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type = str, default = paths.FILE_DEFAULT,
                        help='File to read')
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
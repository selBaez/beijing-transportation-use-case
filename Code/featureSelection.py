"""
This module implements feature selection for commuters classification.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, f_classif

import paths, shared

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    data = pd.read_csv(fileName, index_col='ID')
    print(len(data.index), " records loaded")

    return data

def correlationAnalysis():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.PREPROCESSED_FILE_DEFAULT+'_labeled.csv')

    print("------------------- Separate dataset  per class -------------------")
    commutersData = data[data['LABEL'] == 1]
    non_commutersData = data[data['LABEL'] == 0]

    print("-------------------- Correlation of attributes --------------------")
    attributes = data.select_dtypes(include=[np.number]).columns.values.tolist()
    n_attributes = len(attributes)

    correlationsMatrix = data.corr()
    commuters_correlationsMatrix = commutersData.corr()
    non_commuters_correlationsMatrix = non_commutersData.corr()

    if FLAGS.plot_distr == 'True':
    print("               ----- Plot correlation heatmap -----                ")
        shared._matrixHeatmap('General Correlation', correlationsMatrix, n_attributes, attributes)
        shared._matrixHeatmap('Commuter Correlation', commuters_correlationsMatrix, n_attributes, attributes)
        shared._matrixHeatmap('NonCommuter Correlation', non_commuters_correlationsMatrix, n_attributes, attributes)

    print("---------------------- Feature  importance ------------------------")
    # Exclude card code
    samples = data[attributes[1:-1]].values
    labels = data[attributes[-1]].values

    model = ExtraTreesClassifier()
    model.fit(samples, labels)

    if FLAGS.plot_distr == 'True':
        shared._featureBar('Feature Importance', model.feature_importances_, n_attributes-2, attributes[1:-1])

    print("------------------------ Chi squared  test ------------------------")
    chi2val, pval = chi2(samples, labels)

    if FLAGS.plot_distr == 'True':
        shared._featureBar('Chi squared test', chi2val, n_attributes-2, attributes[1:-1])
        shared._featureBar('p value by chi squared', pval, n_attributes-2, attributes[1:-1])

    print("----------------------------- ANOVA-F -----------------------------")
    fval, pval2 = f_classif(samples, labels)

    if FLAGS.plot_distr == 'True':
        shared._featureBar('F values', fval, n_attributes-2, attributes[1:-1])
        shared._featureBar('p value by ANOVA', pval2, n_attributes-2, attributes[1:-1])


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
    correlationAnalysis()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_distr', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

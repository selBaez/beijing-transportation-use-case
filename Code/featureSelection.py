"""
This module implements feature selection for commuters classification via correlation tests
"""
import argparse, glob
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

import paths, shared

def _loadData(directory):
    """
    Load csv data on pandas
    """
    # Read many sample files
    allFiles = glob.glob(directory + "/*.csv")
    data = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col='ID', header=0)
        list_.append(df)
    data = pd.concat(list_)

    print("{} records loaded").format(len(data.index))

    return data

def _correlationAnalysis(data, columns, n_columns, attributes, n_attributes):
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
        print("Plot")
        shared._correlationHeatmap(correlationsMatrix, n_columns, columns, 'General')
        shared._correlationHeatmap(commuters_correlationsMatrix, n_columns, columns, 'Commuter')
        shared._correlationHeatmap(non_commuters_correlationsMatrix, n_columns, columns, 'Non Commuter')

        shared._featureBar(scores, n_attributes, attributes, 'Correlation', 'to label')

    # TODO: sets of attributes that overlap a lot?

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _featureImportance(samples, labels, attributes, n_attributes):
    """
    Measure feature importance by using a Tree
    """
    model = ExtraTreesClassifier(n_estimators=500, random_state=0)
    model.fit(samples, labels)

    if FLAGS.plot == 'True':
        print("Plot")
        shared._featureBar(model.feature_importances_, n_attributes, attributes, 'Feature Importance', 'scores')

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
        print("Plot")
        shared._featureBar(chi2val, n_attributes, attributes, 'Chi squared', 'scores')
        shared._featureBar(pval, n_attributes, attributes, 'Chi squared', 'p values')

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
        print("Plot")
        shared._featureBar(fval, n_attributes, attributes, 'F values', 'scores')
        shared._featureBar(pval2, n_attributes, attributes, 'F values', 'p values')

    scores = fval

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _domainKnowledge(attributes, n_attributes):
    """
    Read scores given by domain expert
    """
    scores = np.loadtxt(paths.SCORES_DIR_DEFAULT+'domainScores.txt')

    if FLAGS.plot == 'True':
        print("Plot")
        shared._featureBar(scores, n_attributes, attributes, 'Domain knowledge', 'scores')

    # Normalize
    total = np.nansum(scores)
    scores = scores/total

    return scores

def _analysis(data, samples, labels, columns, n_columns, attributes, n_attributes):
    """
    Run statistical, machine learning and domain knowledge analysis to score attributes
    """
    print("Correlation")
    scores_cr = _correlationAnalysis(data, columns, n_columns, attributes, n_attributes)

    print("Feature importance")
    scores_fi = _featureImportance(samples, labels, attributes, n_attributes)

    print("ANOVA-F")
    scores_fv = _anova(samples, labels, attributes, n_attributes)

    print("Domain  knowledge")
    scores_do = _domainKnowledge(attributes, n_attributes)

    # print("Chi squared test")
    # Not using because it unbalances scores towards larger values
    # scores_c2 = _chi2(samples, labels, attributes, n_attributes)

    print("Aggregated scores")
    methods = ['correlation', 'extraTrees importance', 'ANOVA f-value', 'domain knowledge']#, 'chi2']
    scores = [scores_cr, scores_fi, scores_fv, scores_do]#, scores_c2]
    aggScores = np.zeros(shape=scores[0].shape)

    # Normalize scores by method and merge
    for i, values in enumerate(scores):
        scores[i] = values / len(scores)
        aggScores = aggScores + scores[i]

    if FLAGS.plot == 'True':
        print("Plot")
        shared._stackedFeatureBar(scores, methods, n_attributes, attributes, 'Aggregated', 'scores')
        shared._featureBar(aggScores, n_attributes, attributes, 'Merged', 'scores')

    return aggScores

def _selectBest(scores, attributes):
    """
    Select best attributes per category according to their scores
    """
    #TODO change according to new position of num trips

    # Category: General
    generalScores = scores[0:7]
    print("General attributes: {}".format(len(generalScores)))

    k = 2
    selectedGeneral = np.argsort(generalScores)[-k:]
    print("Best {}: {}".format(k, np.array(attributes)[selectedGeneral]))

    # Category: Temporal
    temporalScores = scores[7:9]
    print("\nTemporal attributes: {}".format(len(temporalScores)))

    k = 2
    selectedTemporal = np.argsort(temporalScores)[-k:] + 7
    print("Best {}: {}".format(k, np.array(attributes)[selectedTemporal]))

    # Category: Spatial
    spatialScores = scores[10:]
    print("\nSpatial attributes: {}".format(len(spatialScores)))

    # Calculate scores per spatial attribute ON/OFF pair
    pairSpatialScores = np.zeros(len(spatialScores/2))
    for i in range(len(spatialScores)):
        if i % 2 == 0: # ON
            pairSpatialScores[i/2] = spatialScores[i] + spatialScores[i+1]

    k = 2
    selectedSpatial = np.argsort(pairSpatialScores)[-k:]
    selectedSpatial = np.sort(np.concatenate((selectedSpatial*2+10, selectedSpatial*2+10+1)))
    print("Best {}: {}".format(k*2, np.array(attributes)[selectedSpatial]))

    # Join all
    selected = np.sort(np.concatenate((selectedGeneral, selectedTemporal, selectedSpatial)))
    print("\nFinal selection of {}: {}".format(len(selected), np.array(attributes)[selected]))

    return selected

def selectFeatures():
    """
    Performs training and reports evaluation (on training and validation sets)
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.PREPROCESSED_DIR_DEFAULT+'labeled/'+FLAGS.directory)

    print("----------------------- Extract  attributes -----------------------")
    columns = data.select_dtypes(include=[np.number]).columns.values.tolist()
    n_columns = len(columns)

    # Exclude card code and label
    attributes = columns[1:-1]
    n_attributes = len(attributes)

    # Get values for samples and labels
    samples = data[attributes].values
    labels = data[columns[-1]].values

    print("-------------------------- Run  analysis --------------------------")
    scores = _analysis(data, samples, labels, columns, n_columns, attributes, n_attributes)

    print("--------------- Select best attributes per category ---------------")
    selected = _selectBest(scores, attributes)

    print("--------------------------- Load  cubes ---------------------------")
    # # Load label, std cubes
    # labelDirectory = 'labeled/'
    # stdDirectory = 'std/'
    # directory = paths.CUBES_DIR_DEFAULT+labelDirectory+stdDirectory
    #
    # with open(directory+'combined.pkl', 'r') as fp: userStructures = cPickle.load(fp)



    # cv = KFold(2)
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
    parser.add_argument('--directory', type = str, default = 'original',
                        help='Directory to get files: original or std')
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

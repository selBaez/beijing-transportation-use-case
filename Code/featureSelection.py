"""
This module implements feature selection for commuters classification via correlation tests
"""
import argparse, glob
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import cPickle, random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.manifold import TSNE

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
    # Category: General
    generalScores = scores[0:8]
    print("General attributes: {}".format(len(generalScores)))

    k = 3
    selectedGeneral = np.argsort(generalScores)[-k:]
    print("Best {}: {}".format(k, np.array(attributes)[selectedGeneral]))

    # Category: Temporal
    temporalScores = scores[8:10]
    print("\nTemporal attributes: {}".format(len(temporalScores)))

    k = 2
    selectedTemporal = np.argsort(temporalScores)[-k:] + 8
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

def _formatData(selectedSlices):
    """
    Select relevant cube slices and flatten into vectors
    """
    # Load labeled cubes
    with open(paths.CUBES_DIR_DEFAULT+'labeled.pkl', 'r') as fp: userStructures = cPickle.load(fp)

    # Array contains three columns: code, vector, label
    codes = []
    original = []
    features = []
    labels = []

    for code, [cube, label] in userStructures.items():
        # Flat whole
        flatWhole = cube.flatten(order='F')
        # Select slices
        cube = cube[:,:,selectedSlices]
        # Flatten
        flatCube = cube.flatten(order='F')
        # Format
        codes.append(code)
        original.append(flatWhole)
        features.append(flatCube)
        labels.append(label)

    return np.asarray(codes), np.asarray(original), np.asarray(features), np.asarray(labels)

def _visualize(name, features, labels):
    """
    Visualize high dimensional features in low (2) dimensional space
    """
    manifold = TSNE(n_components=2, random_state=0)
    mappedVectors = manifold.fit_transform(features)
    shared._tsneScatter(name, mappedVectors, labels)

def _store(data):
    """
    Store codes, features and labels
    """
    print("Store ", len(data[0])," samples")

    directory = paths.LOWDIM_DIR_DEFAULT
    with open(directory+"supervised.pkl", "w") as fp: cPickle.dump(data, fp)

def selectFeatures():
    """
    Selects features according to different scores, and formats data to feed to classifier
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.PREPROCESSED_DIR_DEFAULT+'labeled')

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
    selectedSlices = _selectBest(scores, attributes)

    print("--------------------- Cube slices  to vectors ---------------------")
    codes, original, features, labels = _formatData(selectedSlices)

    if FLAGS.plot == 'True':
        print("---------------------------- Visualize ----------------------------")
        _visualize('Selected features', features, labels)
        _visualize('Original features', original, labels)

    print("------------------------------ Store ------------------------------")
    _store([codes, features, labels])


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
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

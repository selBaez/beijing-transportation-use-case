"""
This module clusters the manifolds to find common public transit behaviors.
"""
from __future__ import print_function

import argparse, glob
import pandas as pd
import numpy as np
import random, cPickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import paths, shared

############ --- BEGIN default constants --- ############
MAX_CLUSTERS_DEFAULT = 15
RUNS_DEFAULT = 5 # 150 total # Must be a multiple of 5
############ --- END default constants--- ############

def _loadData():
    """
    Load user samples stored as pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'unsupervised.pkl', 'r') as fp: data = cPickle.load(fp)

    return data

def _tune(data):
    """
    Tune number of clusters
    """
    scores = []
    maxClusters = FLAGS.maxClusters
    runs = FLAGS.runs

    train_data, test_data = train_test_split(data, test_size=0.4)

    # Tune number of clusters with silhoutte
    for nClusters in range(2,maxClusters+1):
        clusterer = KMeans(n_clusters = int(nClusters), random_state=0, init='k-means++')

        clusterer.fit(train_data)
        labels =  clusterer.predict(test_data)

        # Calculate silhoutte score
        temp = silhouette_samples(test_data, labels)
        avScore = np.mean(temp)
        stdScore = np.std(temp)
        scores.append([avScore, stdScore])

        print('For clusters: ', nClusters, ', the average silhouette score is: ', avScore)

    scores = np.array(scores)

    if FLAGS.plot == 'True':
        shared._sampleWithStd(range(2,maxClusters+1), scores[:,0], scores[:,0], 'Number of clusters', 'Score', 'Tuning for k')

    # Choose number of clusters with best score
    optimalNClusters = np.argmax(scores[:,0]) + 2
    print('Optimal number of clusters is: ', optimalNClusters)

    return optimalNClusters

def _cluster(optimalNClusters, data):
    """
    Run K means on data
    """
    # TODO: Run several instances of K means to account for random initalization
    clusterer = KMeans(n_clusters = optimalNClusters, random_state=0, init='k-means++')
    labels = clusterer.fit_predict(data)

    return clusterer, data, labels

def _visualize(name, data, labels):
    """
    Visualize data with TSNE, and color by label
    """
    manifold = TSNE(n_components=2, random_state=0)
    features = manifold.fit_transform(data)
    shared._tsneScatter(name, features, labels=labels)

def _loadFrames(directory):
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

    print(len(data.index), "records loaded")

    return data

def _analysis(data, codes, labels):
    """
    Determine intra and inter cluster statistics.
    """
    # Load data
    data = _loadFrames(paths.PREPROCESSED_DIR_DEFAULT+'all')

    # Match code to label
    data['CLUSTER'] = data['CARD_CODE'].replace(to_replace=codes, value=labels)

    # Summary by cluster
    dataPerCluster = data.groupby('CLUSTER')
    print(dataPerCluster['NUM_TRIPS', 'TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_NUM'].mean(), '\n')
    print(dataPerCluster['ON_RINGROAD', 'OFF_RINGROAD', 'ON_MODE', 'OFF_MODE'].mean(), '\n')
    print(dataPerCluster.aggregate({'CARD_CODE': pd.Series.nunique, 'LABEL': len}), '\n')

    # Find percentage of commuters as label from survey
    test = data.drop_duplicates(['LABEL', 'CARD_CODE'])
    cT = pd.crosstab(test['LABEL'], test['CLUSTER'], margins= True)
    print(cT, '\n')

    return data[['CARD_CODE', 'CLUSTER']]

def _store(model, data):
    """
    Store model and card codes with its cluster label
    """
    directory = paths.MODELS_DIR_DEFAULT
    with open(directory+"clusterer.pkl", "w") as fp: cPickle.dump(model, fp)

    data.to_csv(paths.LABELS_DIR_DEFAULT+'clusteredCodes.csv', encoding='utf-8')

def cluster():
    print("---------------------------- Load data ----------------------------")
    [codes, samples] = _loadData()
    print(len(codes), " samples loaded")

    print("----------------------------- Tune  k -----------------------------")
    optimalNClusters = _tune(samples)

    print("----------------------------- Cluster -----------------------------")
    model, samples, labels = _cluster(optimalNClusters, samples)

    print("----------------------- Visualize  clusters -----------------------")
    _visualize('Clustered', samples, labels)

    print("------------------------ Cluster  analysis ------------------------")
    data = _analysis(samples, codes, labels)

    print("------------------------ Save labeled data ------------------------")
    _store(model, data)


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
    cluster()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--maxClusters', type = int, default = MAX_CLUSTERS_DEFAULT,
                        help='Maximum number of clusters to try when tuning.')
    parser.add_argument('--runs', type = int, default = RUNS_DEFAULT,
                        help='Number of time to run each Kmeans.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

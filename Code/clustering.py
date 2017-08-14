"""
This module clusters the manifolds to find common public transit behaviors.
"""
from __future__ import print_function

import argparse
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
    clusterer = KMeans(n_clusters = optimalNClusters, random_state=0)
    labels = clusterer.fit_predict(data)

    return data, labels


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


def _analysis(data, labels):
    """
    Determine intra and inter cluster statistics.
    """
    # Load data
    data = _loadData(paths.PREPROCESSED_DIR_DEFAULT+'labeled')

    # Match code to label
    #TODO

    # Group by label
    data = list(data.groupby('CLUSTER'))

    for clusterLabel, trips in data:
        trips = list(trips.groupby('CARD_CODE'))

        print(len(trips), ' card codes found')

        # Summary
        print(trips.mean(axis=0))

def _store(data):
    """
    Store pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'.pkl', 'w') as fp: cPickle.dump(encodedData, fp)

def cluster():
    print("---------------------------- Load data ----------------------------")
    [codes, samples] = _loadData()
    print(len(codes), " records loaded")

    print("----------------------------- Tune  k -----------------------------")
    optimalNClusters = _tune(samples)

    print("----------------------------- Cluster -----------------------------")
    samples, labels = _cluster(optimalNClusters, samples)

    print("----------------------- Visualize  clusters -----------------------")
    _visualize('Clustered', samples, labels)

    print("------------------------ Cluster  analysis ------------------------")
    _analysis(samples, labels)

    print("------------------------ Save labeled data ------------------------")
    # _store(data)


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

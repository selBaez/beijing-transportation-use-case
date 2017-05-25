"""
This module clusters the manifolds to find common public transit behaviors.
"""
import argparse
import numpy as np
import random, cPickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhoutte_score

import paths, shared

############ --- BEGIN default constants --- ############
MAX_CLUSTERS_DEFAULT = 3 # 15 total
RUNS_DEFAULT = 5 # 150 total # Must be a multiple of 5
############ --- END default constants--- ############

def _loadData():
    """
    Load user samples stored as pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'.pkl', 'r') as fp: data = cPickle.load(fp)

    return data

def _cluster(data):
    """
    Run K means on data
    """
    scores = []
    maxClusters = FLAGS.maxClusters
    runs = FLAGS.runs

    # Tune number of clusters with silhoutte
    for nClusters in range(1,maxClusters):
        clusterer = KMeans(n_clusters = nClusters, random_state=0, init='k-means++')

        labels = clusterer.fit_predict(data)

        # Calculate silhoutte score
        score = silhoutte_score(data, labels)
        print('For clusters: ', nClusters, ', the average silhoutte score is: ', score)
        scores.append(score)

    # Choose number of clusters with best score
    optimalNClusters = np.argmax(scores)

    # TODO: Run several instances of K means to account for random initalization
    clusterer = KMeans(n_clusters = optimalNClusters, random_state=0)
    labels = clusterer.fit_predict(data)

    return data, labels

def _visualize(data, labels):
    """
    Visualize data with TSNE, and color by label
    """
    manifold = TSNE(n_components=2, random_state=0)
    features = manifold.fit_transform(samples)
    shared._lowDimFeaturesScatter('Clustered', features, labels=labels)


def _analysis(data):
    """
    Determine intra and inter cluster statistics.
    """
    # Some plots



def _store(data):
    """
    Store pickle
    """
    with open(paths.LOWDIM_DIR_DEFAULT+'.pkl', 'w') as fp: cPickle.dump(encodedData, fp)

def cluster():
    print("---------------------------- Load data ----------------------------")
    data = _loadData() # standardized, no label cubes

    print("----------------------------- Cluster -----------------------------")
    data, labels = _cluster(data)

    print("----------------------- Visualize  clusters -----------------------")
    _visualize(data, labels)

    print("------------------------ Cluster  analysis ------------------------")
    _analysis(data, labels)

    print("------------------------ Save labeled data ------------------------")
    _store(data)


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
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse route details.')
    parser.add_argument('--plot_distr', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--scriptMode', type = str, default = 'long',
                        help='Run with long  or short dataset.')
    parser.add_argument('--maxClusters', type = int, default = MAX_CLUSTERS_DEFAULT,
                        help='Maximum number of clusters to try when tuning.')
    parser.add_argument('--runs', type = int, default = RUNS_DEFAULT,
                        help='Number of time to run each Kmeans.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)

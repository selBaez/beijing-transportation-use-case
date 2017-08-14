import numpy as np
import shared, paths

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.manifold import TSNE



# Plot fake cube slice
# slices = np.loadtxt('../Data/sets/cubes/Fake/fakeTurist.txt')
# shared._featureSliceHeatmap(slices, 'Day', 'Unknown', '000000')

# Plot sequential vs parallel times
# times = np.loadtxt('Logs/preparation/parallelTimes.txt')
# shared._sampleCalculateStd(times[:,0], times[:,1], 'Number of cores', 'Time (s)', 'Parallel implementation')

# Plot volume of data
# volume = np.loadtxt('Logs/preparation/dataVolume.txt')
# shared._volumeBar(volume, 'Daily volume of records')
# total = np.sum(volume[:,2])
# print(total)

# Plot cummulative vocabulary
# vocabularyFile = np.loadtxt('Logs/preparation/cummulativeVocabulary.txt')
# shared._vocabCum(vocabularyFile[:,:-1], 'Cummulative lines vocabulary size')
# shared._vocabCum(vocabularyFile[:,[0,2]], 'Cummulative stops vocabulary size')

# Plot tsne of samples from Tu
# # Read data
# dataCom = np.loadtxt(paths.PREVIOUS_DIR_DEFAULT+'Commuter_processed_data.txt')[:, :-1]
# dataNonCom = np.loadtxt(paths.PREVIOUS_DIR_DEFAULT+'NonCommuter_processed_data.txt')[:, :-1]
# # Create labels
# labelsCom = np.ones(dataCom.shape[0])
# labelsNonCom = np.zeros(dataNonCom.shape[0])
# # Concatenate
# data = np.concatenate((dataCom, dataNonCom))
# labels = np.concatenate((labelsCom, labelsNonCom))
# # Plot
# manifold = TSNE(n_components=2, random_state=0)
# features = manifold.fit_transform(data)
# shared._tsneScatter("Tu's features", features, labels=labels)

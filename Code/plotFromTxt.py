import numpy as np
import shared, paths

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os



# Plot fake cube slice
# slices = np.loadtxt('../Data/sets/cubes/Fake/fakeTurist.txt')
# shared._featureSliceHeatmap(slices, 'Day', 'Unknown', '000000')

# Plot sequential vs parallel times
# times = np.loadtxt('Logs/preparation/parallelTimes.txt')
# shared._sampleCalculateStd(times[:,0], times[:,1], 'Number of cores', 'Time (s)', 'Parallel implementation')

# Plot volume of data
volume = np.loadtxt('Logs/preparation/dataVolume.txt')
# shared._volumeBar(volume, 'Daily volume of records')
# total = np.sum(volume[:,2])
# print(total)

# Plot cummulative vocabulary
# vocabularyFile = np.loadtxt('Logs/preparation/cummulativeVocabulary.txt')
# shared._vocabCum(vocabularyFile[:,:-1], 'Cummulative lines vocabulary size')
# shared._vocabCum(vocabularyFile[:,[0,2]], 'Cummulative stops vocabulary size')

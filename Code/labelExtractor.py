"""
This module extracts the card codes for commuters and non-commuters from given data structures.
"""

import numpy as np

############ --- BEGIN default directories --- ############
LOAD_DIR_DEFAULT = './Previous work/commuting-classifier/'
SAVE_DIR_DEFAULT = '../Data/labels/'
############ --- END default directories--- ############

commuters = np.loadtxt(LOAD_DIR_DEFAULT+'Commuter_processed_data.txt')
non_commuters = np.loadtxt(LOAD_DIR_DEFAULT+'NonCommuter_processed_data.txt')


np.savetxt(SAVE_DIR_DEFAULT+'commuterCardCodes.txt', commuters[:,0], '%5.0f')
np.savetxt(SAVE_DIR_DEFAULT+'nonCommuterCardCodes.txt', non_commuters[:,0], '%5.0f')

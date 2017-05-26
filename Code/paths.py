"""
This module Contains the file names and directories where to read and write.
"""

############ --- BEGIN default directories --- ############
# File name
FILE_DEFAULT = 'xaa'

# Main directories
RAW_FILE_DEFAULT = '../Data/sets/raw/'+FILE_DEFAULT+'.csv'
CLEAN_FILE_DEFAULT = '../Data/sets/clean/'+FILE_DEFAULT
PREPROCESSED_FILE_DEFAULT = '../Data/sets/preprocessed/'+FILE_DEFAULT

# Vocabularies
VOC_DIR_DEFAULT = '../Data/vocabularies/fullVocabulary'

# Cubes
CUBES_DIR_DEFAULT = '../Data/sets/cubes/'

# Low dimensionality representation
LOWDIM_DIR_DEFAULT = '../Data/sets/lowDim/'

# Plots
PLOT_DIR_DEFAULT = '../Code/Plots/'

# TensorBoard logs
LOG_DIR_DEFAULT = '../Code/Logs/'
STAT_DIR_DEFAULT = LOG_DIR_DEFAULT+'dayStat/'

# Resources
LABELS_DIR_DEFAULT = '../Data/labels/'
PREVIOUS_DIR_DEFAULT = './Previous work/commuting-classifier/'
############ --- END default directories--- ############

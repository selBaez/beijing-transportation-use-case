"""
This module extracts the card codes for commuters and non-commuters from given data structures.
"""

import numpy as np
import pandas as pd
import paths


def _cleanCode(code):
    """ Remove noise in card code from previous structure
    """
    code = str(code)
    cleanCode = code[-6:]

    return int(cleanCode)


# Read data
dataCom = pd.read_csv(paths.PREVIOUS_DIR_DEFAULT+'Commuter_processed_data.txt', sep=" ", header = None)
dataNonCom = pd.read_csv(paths.PREVIOUS_DIR_DEFAULT+'NonCommuter_processed_data.txt', sep=" ", header = None)

# Get only codes
dataCom = dataCom[0]
dataNonCom = dataNonCom[0]

# Clean codes
dataCom = dataCom.apply(lambda x: _cleanCode(x))
dataNonCom = dataNonCom.apply(lambda x: _cleanCode(x))

# Save
np.savetxt(paths.LABELS_DIR_DEFAULT+'commuterCardCodes.txt', dataCom.values, '%5.0f')
np.savetxt(paths.LABELS_DIR_DEFAULT+'nonCommuterCardCodes.txt', dataNonCom.values, '%5.0f')

"""
This module builds a CNN and runs the user cubes through it to engineer features for clustering. 
"""
import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random, cPickle, re, json
from collections import OrderedDict

import paths, shared

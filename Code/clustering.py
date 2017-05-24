"""
This module clusters the manifolds to find common public transit behaviors.
"""
import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random, cPickle, re, json
from collections import OrderedDict

import paths, shared

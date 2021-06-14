# Imports for incremental learner
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeRegressor
from src.src.evaluate_prequential import EvaluatePrequential
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

# Imports for static Learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from time import perf_counter as pc_timer
from functools import wraps
from os import walk

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from pandas.core.common import SettingWithCopyWarning
from collections import Counter

# For significance tests
from scipy.stats import normaltest
from scipy import stats
from math import sqrt

# pd.set_option('display.max_colwidth', 500)
# General Imports
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

import numpy as np
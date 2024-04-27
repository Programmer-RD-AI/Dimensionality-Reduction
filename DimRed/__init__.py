from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pprint import pprint
import json
import wandb
from sklearn.model_selection import ParameterGrid
# TODO make_pipeline is excessive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import threading
import cupy as cp
import warnings
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.pipeline import Pipeline, make_pipeline
from typing import *
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import math
import os
import random
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import datasets
from wandb.lightgbm import wandb_callback, log_summary
import logging
from wandb.xgboost import WandbCallback
load_dotenv()
PROJECT_NAME = os.getenv("PROJECT_NAME")
logging.getLogger('lightgbm').setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
from DimRed.config import *
from DimRed.helper_functions import *
from DimRed.analysis import *
from DimRed.variation import *
from DimRed.evaluation import *

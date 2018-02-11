import os
import sys
import platform
import codecs
import pandas as pd
import numpy as np
import re
import gc
import math
import string
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p', level=logging.INFO)

from tqdm import tqdm
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import Geohash

class Config():
    data_dir = "/mnt/data/mobike"
    cache_dir = data_dir + "/cache"

    ## 原始数据
    train_path = data_dir + "/data/train.csv"
    test_path = data_dir + "/data/test.csv"

    flag = True
    dump = False

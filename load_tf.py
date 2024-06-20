from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import tensorflow as tf
import os

log_dir = 'log/neurips/mctgraph/'

for f in os.walk(log_dir):
    for e in tf.compat.v1.train.summary_iterator(f):
        print(e)
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:58:29 2016

@author: lwl
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
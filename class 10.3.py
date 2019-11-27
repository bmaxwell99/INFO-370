# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:30:04 2019

@author: dark_
"""

import numpy as np

test = np.arange(20)
test = test.reshape((4,5))
third_column = test[:,2]
test[3,] = np.arange(1,6)


r = np.random.binomial(10, .9, size= (5,5))

np.random.binomial()

s = np.log(1 + r)
s

import pandas as pd


pd.Series([1,2,7,8])

pd.Series([1,2,7,8], index = ['wa', 'tx', 'fl', 'mi'])

pd.Series([100,750,30], index = ['Seattle', 'Portland', 'Vancouver'])


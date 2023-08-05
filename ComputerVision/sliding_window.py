# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:02:18 2023

@author: tarak
"""

from learntools.core import binder
binder.bind(globals())

import tensorflow as tf
import matplotlib.pyplot as plt


plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
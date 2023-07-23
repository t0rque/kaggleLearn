#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:18:35 2023

@author: tarak
"""

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')



concrete = pd.read_csv('./concrete.csv')

df = concrete.copy()
## Data Prep
df_train = df.sample(frac = 0.7, random_state = 0)
df_valid = df.drop(df_train.index)


X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]



#model
    
model = keras.Sequential ([
                layers.BatchNormalization( input_shape=input_shape),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),            
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),            
                layers.Dense(1),]
    )


model.compile(
    optimizer = 'sgd',
    loss = 'mae',
    metrics = ['mae'])


history = model.fit (
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 64,
    epochs = 100,
    verbose = 1,)


history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()


print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))



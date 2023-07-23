#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:08:12 2023

@author: tarak
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')



import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping


spotify = pd.read_csv('./spotify.csv')

X = spotify.copy().dropna()
y = X.pop('track_popularity')

artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat))



# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X,y,groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])


X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

y_train = y_train/100  #Rescale popularity from 0-100 to 0-1
y_valid = y_valid/100

input_shape = [X_train.shape[1]]
print("Input Shape {}".format(input_shape))


#### Model 
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, input_shape=input_shape)])

## Compile  Loss and Optimizer
model.compile (
    optimizer='adam',
    loss='mae')


#Early Stopping for optimized learning and avoding over or underfitting
early_stropping = EarlyStopping(
    min_delta = 0.001,
    patience=5,
    restore_best_weights=True,)

#fit
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid,y_valid),
    batch_size = 512,
    epochs=50,
    callbacks=early_stropping,
    verbose=1)


history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {:0.4f}".format(history_df['val_loss'].min))




#starting plot from epoch 10


history_df = pd.DataFrame(history.history)
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {:0.4f}".format(history_df['val_loss'].min))






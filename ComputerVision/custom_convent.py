# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:38:16 2023

@author: tarak
"""
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.python.client import device_lib


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS']= '1'
    

set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


#load training data

ds_train_ = image_dataset_from_directory(
    './input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
    )

ds_valid_ = image_dataset_from_directory(
    './input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)


## Data Pipe line

def convert_to_float(image,label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image,label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
    
    )


ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
    
    )




## Model

print('Num GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))

device_list = []
for devices in tf.config.experimental.list_physical_devices('GPU'):
    print(devices.name)


for devices in tf.config.get_visible_devices():
    print(devices)
    device_list.append(devices)


from tensorflow.keras.layers.experimental import preprocessing

def model_test(device):
    with tf.device(device.device_type):
        print("testing with ",device)
        model = keras.Sequential([
            layers.InputLayer(input_shape=[128, 128, 3]),
            
            # Data Augmentation
            # ____,
            preprocessing.RandomContrast(factor=0.10),
            preprocessing.RandomFlip(mode='horizontal'),
            preprocessing.RandomRotation(factor=0.10),
            # Block One
            layers.BatchNormalization(renorm=True),
            layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPool2D(),

            # Block Two
            layers.BatchNormalization(renorm=True),
            layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPool2D(),

            # Block Three
            layers.BatchNormalization(renorm=True),
            layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPool2D(),

            # Head
            layers.BatchNormalization(renorm=True),
            layers.Flatten(),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
        
        model.summary()


        ## Fit
        model.compile(
            optimizer = tf.keras.optimizers.Adam(epsilon=0.01),
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
            )
        
        
        print(" ==== USING === ",device)
        history = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs= 40,
            verbose=1)



'''
        import pandas as pd

        history_frame = pd.DataFrame(history.history)
        history_frame.loc[:, ['loss', 'val_loss']].plot()
        history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
'''

results = []
for device in device_list:
    if(device.device_type == "GPU"):
        details = tf.config.experimental.get_device_details(device)
        print(tf.test.gpu_device_name())
        data = details.get('device_name', 'Unkonwn')
        t0 = time.clock()
        model_test(device)
        t1 = time.clock() - t0
        print("Time elapsed: ", t1 - t0) # CPU seconds elapsed (floating point)
        results.append((device,data, t0,t1))

print("============ Modelling time Summary =========")
for result in results:
    print("Time Taken ", result)

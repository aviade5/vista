# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:00:05 2023

@author: user
"""

#sest models
from sklearn.utils import shuffle
import datetime
import pandas as pd
from keras.utils import to_categorical
import cv2
import os
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

import numpy as np
from sklearn.model_selection import train_test_split
import mysql.connector
from PIL import Image
from io import BytesIO


import csv

from sklearn.metrics import classification_report
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.layers import (
    Reshape,Concatenate,Conv1D,TimeDistributed,RepeatVector,Bidirectional,SpatialDropout1D,Embedding,GlobalAveragePooling2D, Input,RandomTranslation,RandomFlip,RandomContrast,RandomRotation,MaxPool2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score

from keras.models import Model

from keras.applications.vgg19  import  VGG19

from keras.callbacks import EarlyStopping


# 2.) parameters
folder_1_path = r"C:\Users\user\Desktop\cities_round_5\1"
folder_0_path = r"C:\Users\user\Desktop\cities_round_5\0"
nRows = 224  # Width
nCols = 224  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1
n_classes = 2
Epochs = 1
n_folds = 10


# 3.) read data
img_1_class = []
lable_1_class = []
img_0_class = []
lable_0_class = []
dic = {}

for filename in os.listdir(folder_1_path):
    
    
    try:
        img_1_class.append(cv2.resize(cv2.imread(folder_1_path+"\\"+filename, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
        dic[filename]= 1
        lable_1_class.append(1)
    except:
        print('error',filename)
        
for filename in os.listdir(folder_0_path):
    #print(filename)
    img_0_class.append(cv2.resize(cv2.imread(folder_0_path+"\\"+filename, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
    dic[filename]= 0
    lable_0_class.append(0)

# 4.) split data
images = img_1_class+img_0_class
lables = lable_1_class+lable_0_class



 # 6.)split the data to train and test
train_x, test_x, train_y, test_y = train_test_split(images, lables, random_state=101,test_size=0.2)
        
# 10.) Convert to Numpy Arrays
train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


Callback = [EarlyStopping(monitor='accuracy', patience = 10)]


base_model = VGG19(input_shape=(224,224,3), include_top=False, weights='imagenet',classes=n_classes)
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = Dense(n_classes, activation='sigmoid')(x)
model = Model(base_model.input, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Callback = [EarlyStopping(monitor='accuracy', patience = 10)]
startTime = datetime.datetime.now()
print("model:{} - Starting Time: {}".format('VGG19',startTime))
history = model.fit(train_x,train_y,epochs=Epochs, callbacks=Callback)
endTrainingTime = datetime.datetime.now()
print("model:{} - ending Time: {}".format('VGG19',endTrainingTime))

predict = model.predict(test_x)    
y_pred = np.argmax(predict, axis=1)
y_max = np.argmax(test_y, axis=1)
print('VVG19',":\n",classification_report(y_max, y_pred))

model.save(r'C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\view_model_round_5.h5')










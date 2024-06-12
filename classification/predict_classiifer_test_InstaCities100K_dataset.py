# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:12:29 2023

@author: user
"""

import cv2
import os
import numpy as np
import random
import csv
from tensorflow.keras.models import load_model

# 2.) parameters
model_path="C:\\Users\\user\\Documents\\עמק יזרעאל\\שנה ג\\work\\ערים\\view_model_round_5.h5"

nRows = 224  # Width
nCols = 224  # Height
cities = ['singapore','sydney','toronto','chicago','london','losangeles','melbourne','miami','newyork','sanfrancisco']

model = load_model(model_path)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


with open('InstaCities100K_dataset_classifier_predictions2.csv', 'w', newline='') as csvfile: #a-append, w-rewrite
     fieldnames = ['City','Image_ID', 'Prediction', 'Rate']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
         
     for city in cities:
        folder_path = "C:\\Users\\user\\Desktop\\world_cities\\{}".format(city)
        images = []
        ids = []
        
        
        for filename in os.listdir(folder_path):
            images.append(cv2.resize(cv2.imread(folder_path+"\\"+filename, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
            ids.append(filename)
        
        z = list(zip(images,ids))
        random.shuffle(z)
        images,ids = zip(*z)
        
        
       
        print('starts',city)    
        for i in range(10000):
            img = np.expand_dims(images[i], axis=0)
            pred = model.predict(img)
            Class = np.argmax(pred)
            
            writer.writerow({'City':city,
                             'Image_ID':ids[i],
                             'Prediction':Class,
                             'Rate':max(max(pred))})
        print('end',city)
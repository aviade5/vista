+# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:19:07 2023

@author: user
"""
import cv2
import os
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import csv
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score
from sklearn.utils import shuffle

# 2.) parameters
folder_1_path = r"C:\Users\user\Desktop\test\1"
folder_0_path = r"C:\Users\user\Desktop\test\0"
nRows = 224  # Width
nCols = 224  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1
nClass = 2

# 3.) read data
img_1_class = []
lable_1_class = []
img_0_class = []
lable_0_class = []
dic = {}

for filename in os.listdir(folder_1_path):
    img_1_class.append(cv2.resize(cv2.imread(folder_1_path+"\\"+filename, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
    dic[filename]= 1
    lable_1_class.append(1)
    
for filename in os.listdir(folder_0_path):
    img_0_class.append(cv2.resize(cv2.imread(folder_0_path+"\\"+filename, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
    dic[filename]= 0
    lable_0_class.append(0)

# 4.) split data
images = img_1_class+img_0_class
lables = lable_1_class+lable_0_class

images,lables = shuffle(images,lables)

test_x = np.array(images)
test_y = to_categorical(lables)

model = load_model(r'C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\view_model_round_5.h5')

predict = model.predict(test_x)    
y_pred = np.argmax(predict, axis=1)
y_max = np.argmax(test_y, axis=1)
print('model_cities_round_1',":\n",classification_report(y_max, y_pred))

with open(r'C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\test\Instagram Top-10 Israeli Cities dataset_Classifier_Performance.csv', 'a', newline='') as csvfile: #a-append, w-rewrite
        fieldnames = ['Model','round_num','Accuracy','Recall','Precision','F1','F1_macro','F1_weighted','n_class','len_of_testset','num_0_class_in_test','num_1_class_in_test']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer.writeheader()
        writer.writerow({'Model':'VVG19',
                         'round_num':5,
                         'Accuracy':accuracy_score(y_max, y_pred),
                         'Recall':recall_score(y_max, y_pred ),
                         'Precision':precision_score(y_max, y_pred),
                         'F1':f1_score(y_max, y_pred),
                         'F1_macro':f1_score(y_max, y_pred,average='macro'),
                         'F1_weighted':f1_score(y_max, y_pred,average='weighted'),
                         'n_class':nClass,
                         'len_of_testset':len(lables),
                         'num_0_class_in_test': len(lable_0_class),
                         'num_1_class_in_test': len(lable_1_class)})
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:24:30 2023

@author: Yarden Aronson
"""

from PIL import Image
import pandas as pd

data = pd.read_csv(r"C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\code\world_test_result.csv")
data2 = pd.read_csv(r"C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\code\world_test_result_2.csv")
data = pd.concat([data,data2],ignore_index=True)

num_of_images=100
city = 'chicago'
images_path= r'C:\Users\user\Desktop\world_cities'
saving_path= r'C:\Users\user\Documents\עמק יזרעאל\שנה ג\work\ערים\code\test_saving'
Prediction=1
rate=0.9
  

def save_images(data,images_path,saving_path="",num_of_images=100,rate=0,Prediction=1,city="all"):
    selected_data = pd.DataFrame()
    if(city!="all"):
        selected_data = data[(data["City"]==city) & (data["Rate"]>=rate) & (data["Prediction"]==Prediction)]
    else:
        selected_data = data[(data["Rate"]>=rate) & (data["Prediction"]==Prediction)]

    selected_data = selected_data.sample(frac=1).reset_index(drop=True)
    
    for i in range(num_of_images):
        img = Image.open("{}\\{}\\{}".format(images_path,selected_data["City"][i],selected_data["Image_ID"][i]))
        img.save("{}\\{}.jpg".format(saving_path,selected_data["Image_ID"][i]))



save_images(data,num_of_images=num_of_images,rate=rate,images_path=images_path,saving_path=saving_path ,Prediction=Prediction,city=city)

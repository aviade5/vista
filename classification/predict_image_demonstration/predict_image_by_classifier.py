#Imports
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('view_model_round_5.h5')

# Set the image dimensions and channels
nRows = 224  # Width
nCols = 224  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1

#Image to predict
img_path = 'city.jpeg'

try:
    # Load and resize the image
    img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC)
    img = np.array(img)
    
except Exception as e:
    print(f"An error occurred while loading and resizing the image: {e}")
    img = None

if img is not None:
    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class of the image
    predict = model.predict(img)
    y_pred = np.argmax(predict, axis=1)

    # Print the predicted class
    print(f'The predicted class is: {y_pred[0]}')
else:
    print("The image could not be processed.")

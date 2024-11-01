# Digit_recognizer_CNN
This project aims to predict handwritten digits from a dataset of tens of thousands of images. Using Convolutional Neural Networks (CNNs), specifically the LeNet-5 architecture, this model is trained to classify handwritten digits with high accuracy.

### Project Overview
Handwritten digit recognition is a fundamental problem in computer vision and deep learning. This project utilizes the LeNet-5 architecture, originally proposed by Yann LeCun, which has been highly effective in identifying characters in handwritten datasets. The goal is to correctly classify each digit from a given image, enhancing our understanding of image processing and deep learning.

### Dataset
The dataset consists of grayscale images of handwritten digits. Each image is of fixed size, with each pixel representing grayscale intensity. The dataset is split into training and testing sets , enabling quick model evaluation. The testing set is used to make predictions and submit those predictions on handwritten digits in Kaggle.

### Data source Link: 
https://www.kaggle.com/competitions/digit-recognizer/data

### Model Architecture: LeNet-5
![image](https://github.com/user-attachments/assets/88cf323f-a613-4d6c-bd7f-804b486fff15)

LeNet-5 is a Convolutional Neural Network (CNN) architecture designed specifically for image classification tasks. Here’s a summary of the layers used in this architecture:

Input Layer: Takes in 32x32 grayscale images, which are scaled to fit the LeNet-5 design.

C1 Convolution Layer: Applies six 5x5 filters, resulting in six feature maps of size 28x28.

S2 Subsampling Layer: Uses average pooling with a stride of 2, reducing the feature maps to size 14x14.

C3 Convolution Layer: Applies sixteen 5x5 filters on the 14x14 feature maps, creating sixteen 10x10 feature maps.

S4 Subsampling Layer: Another average pooling layer with a stride of 2, reducing the maps to size 5x5.

C5 Convolution Layer: Uses 120 filters of size 5x5, outputting a 1x1 map with 120 units.

F6 Fully Connected Layer: A fully connected layer with 84 units.

Output Layer: A softmax layer for classification, which outputs probabilities for each of the ten digits (0–9).

This architecture is highly effective for image classification tasks involving small-scale images and remains foundational in CNN research.

### Packages and Libraries
The following packages were used for implementing, training, and processing data for the model:

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.image import resize

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D

### Key Points
TensorFlow and Keras: Used as the primary deep learning framework to build and train the CNN model.

Sequential: Allows the creation of a linear stack of layers for the CNN.

#### Layers:
Conv2D: Convolutional layer for feature extraction from images.

MaxPooling2D and AveragePooling2D: Pooling layers to down-sample the feature maps.

Flatten: Flattens the pooled feature maps into a one-dimensional vector.

Dense: Fully connected layer for classification.

Dropout: Prevents overfitting by randomly dropping units during training.

#### Additional Tools:
train_test_split: Splits the dataset into training and testing sets.

Pandas and NumPy: Used for data manipulation and efficient handling of large datasets.

img_to_array and resize: Preprocess image data by converting images to arrays and resizing them.

### Results
The model achieves an accuracy of 99.85 on the test the test set, demonstrating the robustness of the LeNet-5 architecture in image recognition tasks.

### Accomplishments
As a testament to my work in image classification and machine learning, I achieved a rank of 744 worldwide out of 1,446 participants in a global competition. This accomplishment underscores the effectiveness of the LeNet-5 architecture and the depth of my experience in this field.

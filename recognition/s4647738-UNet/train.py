# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MnYUCM4kvdRT1U7XTIEhg0nyTIQfBVnV
"""

!pip install kora -q

from kora import drive
drive.link_nbs()

import tensorflow as tf
import dataset 
import modules

def train(xtrain, ytrain, xval, yval):
  """
  Trains model from modules.py
  Parameters:
  xtrain, ytrain, xval, yval: images and their corresponding masks loaded by dataset.py
  Returns:
  model: generated model from training
  history: variable containing all the training statistics (DSC, loss)
  """
  model = modules.improved_unet()
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[modules.dice_similarity])
  history = model.fit(xtrain, ytrain, batch_size=32, epochs=50, validation_data=(xval, yval))
  model.save('improved_unet.model')
  return model, history

# Loading of each ISIC folder
seg_train_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Training_Part1_GroundTruth'
train_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Training_Data'
seg_test_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Test_v2_Part1_GroundTruth'
test_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Test_v2_Data'
#seg_val_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Validation_Part1_GroundTruth'
#val_path = '/content/drive/MyDrive/ISIC/ISIC-2017_Validation_Data'
xtrain, ytrain = dataset.load_dataset(train_path, seg_train_path)
#xval, yval = dataset.load_dataset(val_path, seg_val_path)
xtest, ytest = dataset.load_dataset(test_path, seg_test_path)

model, history = train(xtrain, ytrain, xtest, ytest)

import matplotlib.pyplot as plt

# Plotting DSC for training and validation
plt.plot(history.history['dice_similarity'], label='DSC')
plt.plot(history.history['val_dice_similarity'], label='Validation DSC')
plt.xlabel('Epoch')
plt.ylabel('DSC')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.title("Dice Similarity Coefficient (DSC)")

# Plotting loss for training and validation
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.6])
plt.legend(loc = 'lower right')
plt.title("Loss")
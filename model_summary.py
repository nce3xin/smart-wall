from keras.models import load_model
import numpy as np
import keras
import os 
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model

if __name__=='__main__':
    model_path='ckps/resnet10/weights-036-0.977.hdf5'
    model=load_model(model_path)

    model.summary()

    imgs_root='model/'
    plot_model(model, to_file=imgs_root + 'resnet10_basic_block_(1,1,1,1).pdf')
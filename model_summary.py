from keras.models import load_model
import numpy as np
import keras
import os 
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model

if __name__=='__main__':
    model_path='ckps/resnet_14_bottleneck/weights-041-0.979.hdf5'
    model=load_model(model_path)

    model.summary()

    imgs_root='model/'
    #plot_model(model, to_file=imgs_root + 'resnet16_basic_block_(1,2,2,2).pdf')
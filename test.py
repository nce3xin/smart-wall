from keras.models import load_model
import numpy as np
import keras

def load_(filepath):
    model= load_model(filepath)

if __name__=='__main__':
    model_path='ckps/resnet34/1-fold/weights-169-0.993.hdf5'
    model=load_model(model_path)

    X_file_name='X.npy'
    y_file_name='y.npy'

    X_group1=np.load('./data/test_npy/group1' + '/' + X_file_name)
    y_group1=np.load('./data/test_npy/group1' + '/' + y_file_name)
    X_group1=np.expand_dims(X_group1,axis=3)
    y_group1=keras.utils.to_categorical(y_group1,num_classes=2)

    score=model.evaluate(X_group1,y_group1,batch_size=32)
    print(score)
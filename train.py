from data_loader import data_split
from model import VGG
import keras
import numpy as np

if __name__=='__main__':
    root='./data/npy/'
    X_file_name='X.npy'
    y_file_name='y.npy'

    X_train,X_test,y_train,y_test=data_split(root,X_file_name,y_file_name)

    # 0,1,-1 are invalid values. Replace them with 0.
    X_train=np.where(X_train==1,0,X_train)
    X_train=np.where(X_train==-1,0,X_train)
    X_test=np.where(X_test==1,0,X_test)
    X_test=np.where(X_test==-1,0,X_test)

    X_train=np.expand_dims(X_train,axis=3)
    X_train=X_train.reshape((X_train.shape[0],80,50,1))
    X_test=X_test.reshape((X_test.shape[0],80,50,1))
    '''
    y_train=keras.utils.to_categorical(y_train,num_classes=2)
    y_test=keras.utils.to_categorical(y_test,num_classes=2)
    '''
    model=VGG()
    model.fit(X_train,y_train,batch_size=32,epochs=10)
    score=model.evaluate(X_test,y_test,batch_size=32)
    
    print(score)
from keras.models import load_model
import numpy as np
from data_loader import data_split
import keras
from training_visualization import training_vis
import resnet
import pandas as pd

def _random_sample(array,frac):
    print(array.shape)
    x,y,z=array.shape
    res=pd.DataFrame(array.reshape(x,y*z)).sample(frac=frac, replace=False, random_state=1).values.reshape(-1,y,z)
    print(res.shape)
    return res, res.shape[0]

def _pos_neg_split(X_train,y_train):
    pos_idx=np.squeeze(np.argwhere(y_train==1).T)
    pos=X_train[pos_idx]
    print(y_train.shape)
    y_pos=y_train[pos_idx]
    neg_idx=np.squeeze(np.argwhere(y_train==-1).T)
    neg=X_train[neg_idx]
    y_neg=y_train[neg_idx]
    return pos,neg,y_pos,y_neg

def down_sampling(X_train,y_train):
    pos,neg,y_pos,y_neg=_pos_neg_split(X_train,y_train)
    frac=pos.shape[0]/float(neg.shape[0])
    neg_downsampling,num_after_downsampling =_random_sample(neg,frac)
    y_neg=y_neg[:num_after_downsampling]
    print(y_pos.shape)
    print(y_neg.shape)
    return np.concatenate((pos, neg_downsampling), axis=0), np.concatenate((y_pos,y_neg))

if __name__ == "__main__":
    
    file_path='./ckps/weights-169-0.993.hdf5'
    model= load_model(file_path)
    
    #print(len(model.layers))

    img_channels=1
    img_rows=10
    img_cols=400
    num_classes=2
    
    model=resnet.ResnetBuilder.build_resnet_34((img_channels,img_rows,img_cols),num_classes)
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    root='./data/test_npy/group6/'
    X_file_name='X.npy'
    y_file_name='y.npy'
    X_train_k_fold,X_test_k_fold,y_train_k_fold,y_test_k_fold = data_split(root,X_file_name,y_file_name)

    
    FREEZE_LAYERS=120
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable=False
    

    for i in range(len(X_train_k_fold)):
        X_train=X_train_k_fold[i]
        X_test=X_test_k_fold[i]
        y_train=y_train_k_fold[i]
        y_test=y_test_k_fold[i]

        X_train,y_train=down_sampling(X_train,y_train)
        print('downsampling finished!')
        print(X_train.shape)
        print(y_train.shape)

        # label must start from 0. Convert -1 and +1 to 0 and +1.
        y_train=np.clip(y_train,0,1)
        y_test=np.clip(y_test,0,1)
        
        unique_train, counts_train=np.unique(y_train, return_counts=True)
        print(np.asarray((unique_train, counts_train)).T)
        unique_test, counts_test=np.unique(y_test, return_counts=True)
        print(np.asarray((unique_test, counts_test)).T)

        
        X_train=np.expand_dims(X_train,axis=3)
        X_test=np.expand_dims(X_test,axis=3)
        y_train=keras.utils.to_categorical(y_train,num_classes=2)
        y_test=keras.utils.to_categorical(y_test,num_classes=2)

        history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=32,epochs=200,shuffle=True)

        score=model.evaluate(X_test,y_test,batch_size=32)
        print(score)

        imgs_root='imgs/imgs/transfer_learning/'
        training_vis(history,i+1,imgs_root)
        

        if i==0:
            break
        


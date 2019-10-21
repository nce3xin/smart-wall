from keras.models import load_model
import numpy as np
from data_loader import data_split
import keras
from training_visualization import training_vis
import resnet
import pandas as pd
from keras.callbacks import ModelCheckpoint

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

    
    FREEZE_LAYERS=110
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable=False
    

    for i in range(len(X_train_k_fold)):
        X_train=X_train_k_fold[i]
        X_test=X_test_k_fold[i]
        y_train=y_train_k_fold[i]
        y_test=y_test_k_fold[i]

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

        checkpointer = ModelCheckpoint(filepath='./ckps/transfer_learning/weights-{epoch:03d}-{val_acc:.3f}.hdf5',verbose=1, save_best_only=True,period=1)

        history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=128,epochs=20,shuffle=True,callbacks=[checkpointer])

        score=model.evaluate(X_test,y_test,batch_size=128)
        print(score)

        imgs_root='imgs/imgs/transfer_learning/'
        training_vis(history,i+1,imgs_root)
        

        if i==0:
            break
        


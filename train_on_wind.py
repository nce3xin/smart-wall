from keras.models import load_model
from sklearn.metrics import confusion_matrix
from data_loader import data_split
from training_visualization import training_vis
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
import resnet
import os

if __name__=='__main__':
    model_path='ckps/weights-044-0.976.hdf5'
    model=load_model(model_path)
    
    X_file_name='X.npy'
    y_file_name='y.npy'

    X=np.load('data/wind_data/' + X_file_name)
    y=np.load('data/wind_data/' + y_file_name)

    X_wind_train_k_fold,X_wind_test_k_fold,y_wind_train_k_fold,y_wind_test_k_fold = data_split('data/wind_data/',X_file_name,y_file_name)

    unique_train, counts_train=np.unique(y, return_counts=True)
    print('How many 0 and 1 in y: {}'.format(np.asarray((unique_train, counts_train)).T))
        
    y=keras.utils.to_categorical(y,num_classes=2)
    
    img_channels=1
    img_rows=10
    img_cols=400
    num_classes=2
    
    model=resnet.ResnetBuilder.build_resnet_18((img_channels,img_rows,img_cols),num_classes)
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    #model.summary()
    #plot_model(model, to_file=imgs_root + 'model.pdf')

    checkpointer = ModelCheckpoint(filepath='./ckps/wind/weights-{epoch:03d}-{val_acc:.3f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

    history=model.fit(X,y,batch_size=128,epochs=50,callbacks=[checkpointer])

    imgs_root='imgs/wind/'

    training_vis(history,0,imgs_root)
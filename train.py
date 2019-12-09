from data_loader import data_split
from training_visualization import training_vis
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
import resnet
import os
#import SVC

if __name__=='__main__':
    model_name='ResNet' # ResNet
    imgs_root='imgs/resnet_others_1/'

    root='data/all_plus_wind/'
    X_file_name='X.npy'
    y_file_name='y.npy'

    X_train_k_fold,X_test_k_fold,y_train_k_fold,y_test_k_fold = data_split(root,X_file_name,y_file_name)

    for i in range(len(X_train_k_fold)):
        X_train=X_train_k_fold[i]
        X_test=X_test_k_fold[i]
        y_train=y_train_k_fold[i]
        y_test=y_test_k_fold[i]

        unique_train, counts_train=np.unique(y_train, return_counts=True)
        print('How many 0 and 1 in y_train: {}'.format(np.asarray((unique_train, counts_train)).T))

        unique_test, counts_test=np.unique(y_test, return_counts=True)
        print('How many 0 and 1 in y_test: {}'.format(np.asarray((unique_test, counts_test)).T))
        

        X_train_export_fn='X_train_1_fold.npy'
        if not os.path.exists(root+X_train_export_fn):
            np.save(root+X_train_export_fn,X_train)
        
        X_test_export_fn='X_test_1_fold.npy'
        if not os.path.exists(root+X_test_export_fn):
            np.save(root+X_test_export_fn,X_test)

        y_train_export_fn='y_train_1_fold.npy'
        if not os.path.exists(root+y_train_export_fn):
            np.save(root+y_train_export_fn,y_train)

        y_test_export_fn='y_test_1_fold.npy'
        if not os.path.exists(root+y_test_export_fn):
            np.save(root+y_test_export_fn,y_test)

        print('{} fold -----------------------------------------------------'.format(i+1))

        if model_name=='ResNet':
            print('model: ResNet------------------------------------------------------')
            
            y_train=keras.utils.to_categorical(y_train,num_classes=2)
            y_test=keras.utils.to_categorical(y_test,num_classes=2)
            
            img_channels=1
            img_rows=10
            img_cols=400
            num_classes=2
            
            model=resnet.ResnetBuilder.build_resnet_others_1((img_channels,img_rows,img_cols),num_classes)
            model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

            #model.summary()
            #plot_model(model, to_file=imgs_root + 'model.pdf')

            checkpointer = ModelCheckpoint(filepath='./ckps/resnet_others_1/weights-{epoch:03d}-{val_acc:.3f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

            history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=128,epochs=50,callbacks=[checkpointer])

            # load the model
            #model= load_model(filepath)

            #score=model.evaluate(X_test,y_test,batch_size=32)
            #score=model.evaluate(X_group1,y_group1,batch_size=32)
            #predicted_labels=model.predict(X_group1, batch_size=32)

            training_vis(history,i+1,imgs_root)
            
            #print(score)
            
            if i==0:
                break
            
        '''   
        elif model_name=='SVC':
            print('model: SVC------------------------------------------------------')
            svc=SVC.SVC_model()
            X_train=X_train.reshape((X_train.shape[0],4000))
            X_test=X_test.reshape((X_test.shape[0],4000))
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            svc.fit(X_train,y_train)
            score=svc.score(X_test,y_test)
            print(score)
        '''
            
from data_loader import data_split
import keras
import numpy as np
import resnet

if __name__=='__main__':
    root='./data/npy/'
    X_file_name='X.npy'
    y_file_name='y.npy'

    X_train,X_test,y_train,y_test=data_split(root,X_file_name,y_file_name)

    X_train=np.expand_dims(X_train,axis=3)
    X_test=np.expand_dims(X_test,axis=3)
    '''
    X_train=X_train.reshape((X_train.shape[0],80,50,1))
    X_test=X_test.reshape((X_test.shape[0],80,50,1))
    '''
    y_train=keras.utils.to_categorical(y_train,num_classes=2)
    y_test=keras.utils.to_categorical(y_test,num_classes=2)
    
    img_channels=1
    img_rows=10
    img_cols=400
    num_classes=2
    model=resnet.ResnetBuilder.build_resnet_34((img_channels,img_rows,img_cols),num_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=32,epochs=10)
    score=model.evaluate(X_test,y_test,batch_size=32)
    
    print(score)
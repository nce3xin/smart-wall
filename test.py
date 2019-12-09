from keras.models import load_model
import numpy as np
import keras
import os 
from sklearn.metrics import confusion_matrix

if __name__=='__main__':
    model_path='ckps/weights-044-0.976.hdf5'
    model=load_model(model_path)

    #model.summary()
    
    X_file_name='X_test_1_fold.npy'
    y_file_name='y_test_1_fold.npy'

    X=np.load('data/' + X_file_name)
    y=np.load('data/' + y_file_name)
    
    unique_train, counts_train=np.unique(y, return_counts=True)
    print(np.asarray((unique_train, counts_train)).T)

    y_to_categorical=keras.utils.to_categorical(y,num_classes=2)
    '''
    score=model.evaluate(X,y_to_categorical,batch_size=128)
    print('loss & accuracy: {}'.format(score))
    '''
    y_prob=model.predict(X,batch_size=128)
    y_pred = np.argmax(y_prob, axis=1)
    print('y_true shape: {}'.format(y.shape))
    print('y_pred shape: {}'.format(y_pred.shape))
    print(y_prob.shape) # (num_samples,2)
    print(y_prob[:5])

    cm=confusion_matrix(y,y_pred)
    print('confusion matrix: {}'.format(cm))

    y_pred_prob_export_path='data/y_predict_prob.npy'
    if not os.path.exists(y_pred_prob_export_path):
        np.save(y_pred_prob_export_path, y_prob)
    
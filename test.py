from keras.models import load_model
import numpy as np
import keras
import os 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from plot_cm import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


if __name__=='__main__':
    #model_path='ckps/weights-044-0.976.hdf5'
    model_path='ckps/resnet_others_1/weights-036-0.977.hdf5'
    model=load_model(model_path)

    #model.summary()
    
    X_file_name='X_test_1_fold.npy'
    y_file_name='y_test_1_fold.npy'

    X=np.load('data/all_plus_wind/' + X_file_name)
    y=np.load('data/all_plus_wind/' + y_file_name)
    
    unique_train, counts_train=np.unique(y, return_counts=True)
    print(np.asarray((unique_train, counts_train)).T)

    y_to_categorical=keras.utils.to_categorical(y,num_classes=2)
    '''
    score=model.evaluate(X,y_to_categorical,batch_size=128)
    print('loss & accuracy: {}'.format(score))
    '''

    y_pred_export_path='data/all_plus_wind/y_pred.npy'
    if not os.path.exists(y_pred_export_path):
        y_prob=model.predict(X,batch_size=128)
        y_pred = np.argmax(y_prob, axis=1)
        np.save(y_pred_export_path, y_pred)
    else:
        y_pred=np.load(y_pred_export_path)
    
    print('y_true shape: {}'.format(y.shape))
    print('y_pred shape: {}'.format(y_pred.shape))
    #print(y_prob.shape) # (num_samples,2)
    #print(y_prob[:5])
    

    cm=confusion_matrix(y,y_pred)
    print('confusion matrix: {}'.format(cm))
    plot_confusion_matrix(cm)
    '''
    y_pred_prob_export_path='data/y_predict_prob.npy'
    if not os.path.exists(y_pred_prob_export_path):
        np.save(y_pred_prob_export_path, y_prob)
    '''


    # f1 score
    f1=f1_score(y,y_pred)
    print('f1 score: {}'.format(f1))

    # precision
    p=precision_score(y,y_pred)
    print('precision: {}'.format(p))

    # recall
    r=recall_score(y,y_pred)
    print('recall: {}'.format(r))
    
    # auc
    auc=roc_auc_score(y,y_pred)
    print('auc: {}'.format(auc))
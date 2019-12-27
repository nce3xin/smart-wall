from keras.models import load_model
import numpy as np
import keras
import os 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from plot_cm import plot_confusion_matrix
from plot_roc_curve import plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


if __name__=='__main__':
    #model_path='ckps/weights-044-0.976.hdf5'
    model_path='ckps/resnet_17_bottleneck/weights-046-0.979.hdf5'
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

    # Attention! If you change the model architecture, you must delete y_prob.npy and y_pred.npy before testing! 
    '''
    y_prob_export_path='data/all_plus_wind/y_prob.npy'
    if not os.path.exists(y_prob_export_path):
        y_prob=model.predict(X,batch_size=128)
        np.save(y_prob_export_path, y_prob)

    y_pred_export_path='data/all_plus_wind/y_pred.npy'
    if not os.path.exists(y_pred_export_path):
        y_prob=model.predict(X,batch_size=128)
        y_pred = np.argmax(y_prob, axis=1)
        np.save(y_pred_export_path, y_pred)
    else:
        y_pred=np.load(y_pred_export_path)
    '''

    y_prob=model.predict(X,batch_size=128)
    y_pred = np.argmax(y_prob, axis=1)

    print('y_true shape: {}'.format(y.shape))
    print('y_pred shape: {}'.format(y_pred.shape))
    #print(y_prob.shape) # (num_samples,2)
    #print(y_prob[:5])
    

    cm=confusion_matrix(y,y_pred)
    print('confusion matrix: {}'.format(cm))
    plot_confusion_matrix(cm)

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
    #y_prob=np.load(y_prob_export_path)
    auc=roc_auc_score(y,y_prob[:,1])
    print('auc: {}'.format(auc))

    # roc curve
    plot_roc_curve(y,y_prob)
    
    
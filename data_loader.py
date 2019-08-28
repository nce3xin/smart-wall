from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def data_split(root,X_file_name,y_file_name):
    X=np.load(root + '/' + X_file_name)
    y=np.load(root + '/' + y_file_name)
    # Randomized CV splitters may return different results for each call of split. 
    # You can make the results identical by setting random_state to an integer.
    # If shuffle=True and random_state is set, then the result of split is determined. If you re-run the program, the results will not change.
    # If shuffle=False and random_state=None, then the result of split is also determined.
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    X_train_k_fold=[]
    y_train_k_fold=[]
    X_test_k_fold=[]
    y_test_k_fold=[]
    for train_index,test_index in skf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_k_fold.append(X_train)
        y_train_k_fold.append(y_train)
        X_test_k_fold.append(X_test)
        y_test_k_fold.append(y_test)
    '''
    for i in range(len(X_train_k_fold)):
        X_train=X_train_k_fold[i]
        y_train=y_train_k_fold[i]
        X_test=X_test_k_fold[i]
        y_test=y_test_k_fold[i]
        export_root='./data/5-fold-npy/'+str(i+1)+'-fold'
        np.save(export_root+'/X_train.npy', X_train)
        np.save(export_root+'/X_test.npy', X_test)
        np.save(export_root+'/y_train.npy', y_train)
        np.save(export_root+'/y_test.npy', y_test)
    '''
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
    return X_train_k_fold,X_test_k_fold,y_train_k_fold,y_test_k_fold

if __name__=='__main__':
    root='./data/npy/'
    X_file_name='X.npy'
    y_file_name='y.npy'

    data_split(root,X_file_name,y_file_name)
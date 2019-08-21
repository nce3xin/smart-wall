from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

def data_split(root,X_file_name,y_file_name):
    X=np.load(root + '/' + X_file_name)
    y=np.load(root + '/' + y_file_name)
    skf = StratifiedKFold(n_splits=5)
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

    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
    return X_train_k_fold,X_test_k_fold,y_train_k_fold,y_test_k_fold
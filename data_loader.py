from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def data_split(root,X_file_name,y_file_name):
    X=np.load(root + X_file_name)
    y=np.load(root + y_file_name)
    # Randomized CV splitters may return different results for each call of split. 
    # You can make the results identical by setting random_state to an integer.
    # If shuffle=True and random_state is set, then the result of split is determined. If you re-run the program, the results will not change.
    # If shuffle=False and random_state=None, then the result of split is also determined.

    # downsampling
    '''
    X,y=down_sampling(X,y)
    print('downsampling finished!')
    print(X.shape)
    print(y.shape)
    '''
    
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
        export_root='./data/wind_data/'
        np.save(export_root+'X_wind_train.npy', X_train)
        np.save(export_root+'X_wind_test.npy', X_test)
        np.save(export_root+'y_wind_train.npy', y_train)
        np.save(export_root+'y_wind_test.npy', y_test)
        if i==0:
            break
    '''
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
    return X_train_k_fold,X_test_k_fold,y_train_k_fold,y_test_k_fold

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

if __name__=='__main__':
    root='./data/all_plus_wind/'
    X_file_name='X.npy'
    y_file_name='y.npy'

    data_split(root,X_file_name,y_file_name)
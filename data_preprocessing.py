import pandas as pd
import numpy as np
import os

def _split_block_data(x):
    total=[]
    for i in x.split('|')[:-1]:
        single_vec=i.split('+')[-1].split(',')
        single_vec=[float(i) for i in single_vec]
        total.append(single_vec)
    return total

def _replace_invalid_data_with_former_or_latter_valid_data(X):
    sample_num,height,width=X.shape
    closest_valid=None
    candidates=[]
    for i in range(sample_num):
        sample=X[i] # 10*400
        for j in range(height):
            seq=sample[j] # length: 400
            for k in range(width):
                if seq[k] != 0 and seq[k] != 1 and seq[k] != -1:
                    valid=seq[k]
                    closest_valid=valid
                    for c in candidates:
                        X[c[0]][c[1]]=np.where(seq==0,valid,seq)
                        X[c[0]][c[1]]=np.where(seq==1,valid,seq)
                        X[c[0]][c[1]]=np.where(seq==-1,valid,seq)
                    X[i][j]=np.where(seq==0,valid,seq)
                    X[i][j]=np.where(seq==1,valid,seq)
                    X[i][j]=np.where(seq==-1,valid,seq)
                    continue
                if k == width - 1: # reach the end
                    if closest_valid != None:
                        X[i][j]=np.where(seq==0,closest_valid,seq)
                        X[i][j]=np.where(seq==1,closest_valid,seq)
                        X[i][j]=np.where(seq==-1,closest_valid,seq)
                    else:
                        candidates.append([i,j])
    return X

def _check(X): # check 0,-1,1 exist or not
    if 0 in X:
        return False
    if 1 in X:
        return False
    if -1 in X:
        return False
    return True

def _minus_mean(X):
    return X-X.mean(axis=1,keepdims=True)

if __name__=='__main__':
    root='data/all_plus_wind/'
    #data_name='pos_neg_downsampled_concat.csv'
    data_name='all_plus_wind.csv'

    print('*** Start to read {} file...'.format(root+data_name))
    df=pd.read_csv(root + data_name)
    print('*** Reading finished!')

    img_raw=df['block_data']
    label=df['sample_type']

    X=[]
    for x in img_raw.values.tolist():
        img=_split_block_data(x)
        X.append(img)
    X=np.array(X) # X.shape: (16675,10,400)
    y=label.values

    print('*** Start to replace invalid data with former or latter valid data...')
    X=_replace_invalid_data_with_former_or_latter_valid_data(X)
    assert _check(X)==True, 'There exists 0, 1 and -1 that are not filled.'
    print('*** Replacing finished!')

    print('*** Start to minus mean...')
    X=_minus_mean(X)
    print('*** Minus finished!')

    X=np.expand_dims(X,axis=3)
    y=np.clip(y,0,1)

    export_X_fn='X.npy'
    export_y_fn='y.npy'

    if os.path.exists(root+export_X_fn):
        os.remove(root+export_X_fn)
    if os.path.exists(root+export_y_fn):
        os.remove(root+export_y_fn)

    print('*** Start to write X & y to .npy file...')
    np.save(root+export_X_fn,X)
    np.save(root+export_y_fn,y)
    print('*** Writing finished!')
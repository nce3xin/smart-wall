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

if __name__=='__main__':
    root='./data/test_csv/group10/'
    group_id='group10'
    data_name=group_id+'.csv'
    #subdir='CASNeg20190803'
    #data_name=group_id+'_'+ subdir +'.csv'
    df=pd.read_csv(root + data_name, index_col=0)

    img_raw=df['block_data']
    label=df['sample_type']

    X=[]
    for x in img_raw.values.tolist():
        img=_split_block_data(x)
        X.append(img)
    X=np.array(X) # X.shape: (16675,10,400)
    y=label.values

    X=_replace_invalid_data_with_former_or_latter_valid_data(X)

    check=_check(X)
    print(check)

    #pd.DataFrame(X.reshape((X.shape[0],4000))).to_csv('data/X.csv',header=None,index=None)
    '''
    if not os.path.isfile('data/npy/X.npy'):
        np.save('data/npy/X.npy',X)
    if not os.path.isfile('data/npy/y.npy'):
        np.save('data/npy/y.npy',y)
    '''

    np.save('data/test_npy/' + group_id + '/X.npy', X)
    np.save('data/test_npy/' + group_id + '/y.npy', y)
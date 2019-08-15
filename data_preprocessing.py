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

if __name__=='__main__':
    root='./data/'
    data_name='all.csv'
    df=pd.read_csv(root + data_name, index_col=0)

    img_raw=df['block_data']
    label=df['sample_type']

    X=[]
    for x in img_raw.values.tolist():
        img=_split_block_data(x)
        X.append(img)
    X=np.array(X) # X.shape: (16675,10,400)
    y=label.values

    if not os.path.isfile('data/npy/X.npy'):
        np.save('data/npy/X.npy',X)
    if not os.path.isfile('data/npy/y.npy'):
        np.save('data/npy/y.npy',y)
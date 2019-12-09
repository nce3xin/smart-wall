import pandas as pd
import random
import os

def down_sampling(root_dir,filename,export_fn='neg_downsampling.csv'):
    if os.path.exists(root_dir+export_fn):
        os.remove(root_dir+export_fn)
    num_reserved=100000
    total_neg=829141
    print('*** Start to generate skip index...')
    skip_idx=random.sample(range(1,total_neg),total_neg-num_reserved)
    print('*** Skip index generation finished!')
    print('*** Start to read csv with skiprows...')
    df=pd.read_csv(root_dir+filename,skiprows=skip_idx)
    print('*** Reading finished!')
    print('*** Start to write downsampling neg dataframe to .csv file...')
    df.to_csv(root_dir+export_fn,index=False)
    print('*** Write finished!')

if __name__=='__main__':
    root_dir='data/6group_csv/'
    filename='neg.csv'
    down_sampling(root_dir,filename)
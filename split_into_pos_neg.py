import pandas as pd
import os

def num_pos_neg(root_dir,filename):
    reader=pd.read_csv(root_dir+filename,chunksize=5e4)
    total_cnt=0
    total_pos=0
    total_neg=0
    for j,chunk in enumerate(reader):
        print('*** Start to process No.{} chunk.'.format(j+1))
        total_cnt+=chunk.shape[0]
        total_pos+=sum(chunk['sample_type']==1)
        total_neg+=sum(chunk['sample_type']==-1)
    assert total_cnt==total_pos+total_neg, 'total_cnt!=total_pos+total_neg'
    print('Total cnt: {}. Total pos: {}. Total neg: {}.'.format(total_cnt,total_pos,total_neg))
    return total_cnt,total_pos,total_neg

def split_into_pos_neg(root_dir,filename,chunksize=5e4):
    reader=pd.read_csv(root_dir+filename,chunksize=chunksize)
    export_pos_filename='pos.csv'
    export_neg_filename='neg.csv'

    if os.path.exists(root_dir+export_pos_filename):
        os.remove(root_dir+export_pos_filename)
    if os.path.exists(root_dir+export_neg_filename):
        os.remove(root_dir+export_neg_filename)

    for j,chunk in enumerate(reader):
        print('*** Start to process No.{} chunk. Chunksize: {}.'.format(j+1,chunk.shape[0]))

        if j==0:
            pos=chunk.loc[chunk['sample_type']==1]
            neg=chunk.loc[chunk['sample_type']==-1]
        else:
            pos=pos.append(chunk.loc[chunk['sample_type']==1],ignore_index=True)
            neg=neg.append(chunk.loc[chunk['sample_type']==-1],ignore_index=True)

        if pos.shape[0]>chunksize:
            if not os.path.exists(root_dir+export_pos_filename):
                pos.to_csv(root_dir+export_pos_filename,index=False)
            else:
                with open(root_dir+export_pos_filename,'a') as f:
                    pos.to_csv(f,index=False,header=False)
            # empty the pos dataframe
            pos=pd.DataFrame(columns=pos.columns)

        if neg.shape[0]>chunksize:
            if not os.path.exists(root_dir+export_neg_filename):
                neg.to_csv(root_dir+export_neg_filename,index=False)
            else:
                with open(root_dir+export_neg_filename,'a') as f:
                    neg.to_csv(f,index=False,header=False)
            # empty the neg dataframe
            neg=pd.DataFrame(columns=neg.columns)

    if pos.shape[0]!=0:
        print('Last writing for pos. Pos cnt: {}'.format(pos.shape[0]))
        with open(root_dir+export_pos_filename,'a') as f:
            pos.to_csv(f,index=False,header=False)
    if neg.shape[0]!=0:
        print('Last writing for neg. Neg cnt: {}'.format(neg.shape[0]))
        with open(root_dir+export_neg_filename,'a') as f:
            neg.to_csv(f,index=False,header=False)


if __name__=='__main__':
    root_dir='data/6group_csv/'
    filename='data.csv'
    #total_cnt,total_pos,total_neg=num_pos_neg(root_dir,filename)
    # Total cnt: 929445. Total pos: 100304. Total neg: 829141.
    
    #split_into_pos_neg(root_dir,filename)
    total_cnt,total_pos,total_neg=num_pos_neg(root_dir,'neg_downsampling.csv')
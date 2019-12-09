import pandas as pd
import os
from db2csv import list_files

def concat_csv(root_dir,export_root,export_filename,filenames=None):
    export_fullpath = export_root + export_filename
    if os.path.exists(export_fullpath):
        os.remove(export_fullpath)
    if filenames==None:
        filenames = list_files(root_dir)
    cnt=0
    for i,fn in enumerate(filenames):
        reader=pd.read_csv(root_dir+fn,chunksize=5e4)
        for j,chunk in enumerate(reader):
            print('*** Start to concat No.{}: {}, {} chunk. Valid sample cnt: {}'.format(i+1,root_dir+fn,j+1,chunk.shape[0]))
            cnt+=chunk.shape[0]
            if not os.path.exists(export_fullpath):
                chunk.to_csv(export_fullpath,index=False)
            else:
                with open(export_fullpath,'a') as f:
                    chunk.to_csv(f,index=False,header=False)
    print('*** Total cnt: {}'.format(cnt))

if __name__=='__main__':
    '''
    root_dir='data/6group_csv/'
    export_root='data/6group_csv/'
    export_filename='pos_neg_downsampled_concat.csv'
    filenames=['pos.csv','neg_downsampling.csv']
    concat_csv(root_dir,export_root,export_filename,filenames)
    '''
    root_dir='data/6group_csv/'
    export_root='data/all_plus_wind/'
    export_filename='all_plus_wind.csv'
    filenames=['pos_neg_downsampled_concat.csv','group7.csv']
    concat_csv(root_dir,export_root,export_filename,filenames)


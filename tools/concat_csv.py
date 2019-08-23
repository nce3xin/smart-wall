import pandas as pd
import os

def concat_csv(root,filenames,export_file_name):
    file_paths=[root + '/' + f for f in filenames]
    combined_csv = pd.concat( [ pd.read_csv(f) for f in file_paths ], ignore_index=True)
    combined_csv=combined_csv.drop('id',1)
    combined_csv.to_csv( export_file_name + ".csv", index=True )

def _list_files(root,dir):
    filenames=[f for f in os.listdir(root+'/'+dir)]
    return filenames

if __name__=='__main__':
    root='./data/test_csv'
    dir='group6'
    '''
    pos_dir='CASPos20190601_csv'
    neg_dir='CASNeg20190601_csv'
    pos_csv_filenames=_list_files(root,pos_dir)
    neg_csv_filenames=_list_files(root,neg_dir)
    '''
    csv_filenames=_list_files(root,dir)
    concat_csv(root+'/'+dir,csv_filenames,dir)
    #concat_csv(root+'/'+neg_dir,neg_csv_filenames,'all_neg')
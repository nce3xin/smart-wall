import sqlite3
import pandas as pd
import os

def _file_parser(root_dir,file_name):
    conn=sqlite3.connect(root_dir + file_name)
    c=conn.cursor()
    sample_type=c.execute('select sample_type from T_Sample').fetchall()
    sample_type=[i[0] for i in sample_type]
    block_data=c.execute('select block_data from T_Sample').fetchall()
    block_data=[i[0] for i in block_data]
    data={'block_data':block_data,'sample_type':sample_type}
    df=pd.DataFrame(data)
    conn.close()
    return df

def list_files(root_dir):
    filenames=[f for f in os.listdir(root_dir)]
    return filenames

def _file_OK(filename):
    if 'OK' in filename:
        return True
    else:
        return False

if __name__=='__main__':
    root_dir='data/test_db/group7/'

    export_dir='data/test_single_csv_per_group/'
    export_file_name='group7.csv'
    export_fullpath=export_dir + export_file_name

    if os.path.exists(export_fullpath):
        os.remove(export_fullpath)
    
    filenames=list_files(root_dir)
    cnt=0
    invalid_cnt=0
    for i,fn in enumerate(filenames):
        if not _file_OK(fn):
            os.remove(root_dir+fn)
            print('*** {}: file name invalid. Removed.'.format(root_dir+fn))
            invalid_cnt+=1
            continue
        df=_file_parser(root_dir,fn)
        if not os.path.exists(export_fullpath):
            df.to_csv(export_fullpath,index=False)
        else:
            with open(export_fullpath,'a') as f:
                df.to_csv(f,index=False,header=False)
        cnt+=df.shape[0]
        print('*** Start to process No.{}: {}. Valid sample cnt: {}'.format(i+1,root_dir+fn,df.shape[0]))
    print('*** Total cnt: {}'.format(cnt))
    print('*** Total invalid file cnt: {}'.format(invalid_cnt))
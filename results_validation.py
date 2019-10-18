import numpy as np

def parse_session_log():
    file_path='data/test_npy/group2/'
    file_name='y_predict_prob.npy'
    file_name_log='session.log'
    y_predict_prob=np.load(file_path+file_name)
    
    with open(file_path+file_name_log) as f:
        line=f.readline()
        cnt=0
        while line:
            first_log=float(line.strip().split(',')[1].strip()[1:-1].split(' ')[0])
            second_log=float(line.strip().split(',')[1].strip()[1:-1].split(' ')[1])
            first=y_predict_prob[cnt,0]
            second=y_predict_prob[cnt,1]
            
            if abs(first-first_log)>1e-6 or abs(second-second_log)>1e-6:
                print(cnt)
                print(first,first_log,second,second_log)

            line=f.readline()
            cnt+=1

if __name__=='__main__':
    parse_session_log()
        
import numpy as np

if __name__=='__main__':
    file_path='data/test_y_predict_prob/group2/'
    file_name='y_predict_prob.npy'

    y_predict_prob=np.load(file_path+file_name)
    '''
    print(y_predict_prob.shape)
    print(y_predict_prob[-5:])
    '''
    num_samples=y_predict_prob.shape[0]
    for i in range(num_samples):
        sum=y_predict_prob[i,0]+y_predict_prob[i,1]
        if abs(sum-1.0)>1e-6:
            print(i)
            print(sum)

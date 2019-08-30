import numpy as np
import scipy.io as io

root='data/5-fold-npy/'

for i in range(5):
    folder_name=str(i+1)+'-fold/'
    file_path=root + folder_name

    X_train=np.load(file_path+'X_train.npy')
    X_test=np.load(file_path+'X_test.npy')
    y_train=np.load(file_path+'y_train.npy')

    pos_indice_train=np.where(y_train==1)
    neg_indice_train=np.where(y_train==-1)

    X_train_pos=X_train[pos_indice_train]
    X_train_neg=X_train[neg_indice_train]
    print('# X_train_pos:',X_train_pos.shape)
    print('# X_train_neg:',X_train_neg.shape)

    export_root='data/5-fold-mat/'+str(i+1)+'-fold/'

    # save X_train_pos,X_train_neg,X_test into a single .mat file
    io.savemat(export_root+'data.mat',{'X_train_pos':X_train_pos,'X_train_neg':X_train_neg,'X_test':X_test})
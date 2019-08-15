from sklearn.model_selection import train_test_split
import numpy as np

def data_split(root,X_file_name,y_file_name):
    X=np.load(root + '/' + X_file_name)
    y=np.load(root + '/' + y_file_name)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
    return X_train,X_test,y_train,y_test
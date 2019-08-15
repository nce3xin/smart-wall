from sklearn.model_selection import train_test_split
import numpy as np

if __name__=='__main__':
    root='./data/npy/'
    X=np.load(root+'X.npy')
    y=np.load(root+'y.npy')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
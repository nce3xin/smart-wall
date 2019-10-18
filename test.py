from keras.models import load_model
import numpy as np
import keras

def load_(filepath):
    model= load_model(filepath)

if __name__=='__main__':
    #model_path='ckpt/resnet34/1-fold/weights-169-0.993.hdf5'
    model_path='ckps/weights-169-0.993.hdf5'
    model=load_model(model_path)

    X_file_name='X.npy'
    y_file_name='y.npy'

    group_id='group2'

    X_group1=np.load('./data/test_npy/'+group_id + '/' + X_file_name)
    y_group1=np.load('./data/test_npy/'+group_id + '/' + y_file_name)
    
    unique_train, counts_train=np.unique(y_group1, return_counts=True)
    print(np.asarray((unique_train, counts_train)).T)

    # convert labels -1 and 1 to 0 and 1
    y_group1=np.clip(y_group1,0,1)

    X_group1=np.expand_dims(X_group1,axis=3)
    print(X_group1.shape)
    y_group1=keras.utils.to_categorical(y_group1,num_classes=2)
    '''
    score=model.evaluate(X_group1,y_group1,batch_size=32)
    print(score)
    '''
    y_prob=model.predict(X_group1,batch_size=32)
    print(y_prob.shape) # (num_samples,2)
    print(y_prob[:5])
    #np.save('data/test_y_predict_prob/' + group_id + '/y_predict_prob.npy', y_prob)
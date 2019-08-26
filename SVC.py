from sklearn.svm import SVC

def SVC_model():
    clf = SVC(gamma='auto')
    return clf
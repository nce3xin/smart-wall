import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm):
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True,fmt='.0f',cmap="Blues",xticklabels =['negative','positive'],yticklabels =['negative','positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('ResNet-10 confusion matrix')
    plt.savefig('imgs/resnet_others_1/cm.png',dpi=300)
    print('-----------------confusion matrix figure saved!-------------------')
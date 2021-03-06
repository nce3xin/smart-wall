from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_roc_curve(y,y_prob):
    fpr, tpr, _ = roc_curve(y, y_prob[:,1], pos_label=1)
    auc = roc_auc_score(y, y_prob[:,1])
    plt.figure()
    plt.plot(fpr,tpr,label="ResNet-17-bottleneck ROC curve, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig('imgs/resnet_17_bottleneck/roc_curve.png',dpi=300)
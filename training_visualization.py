import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def training_vis(history,fold,export_root):
    _acc_vis(history,fold,export_root)
    _loss_vis(history,fold,export_root)

def _acc_vis(history,fold,export_root):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(export_root + 'acc_' + str(fold) + '-fold.png',dpi=300)

def _loss_vis(history,fold,export_root):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(export_root + 'loss_' + str(fold) + '-fold.png',dpi=300)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def training_vis(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('imgs/training_vis.png',dpi=300)
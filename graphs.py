import itertools

import matplotlib.pyplot as plt
import os
from datetime import datetime

import numpy as np


class Graph:

    @staticmethod
    def plot_accuracy_loss(model_fit, dir, name):
        plt.plot(model_fit.history['accuracy'], "-", label="accuracy")
        plt.plot(model_fit.history['val_accuracy'], "-", label="val_acc")
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(dir, name + '_accuracy_epochs_graph'+ str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) +'.png'))

        plt.plot(model_fit.history['loss'], "-", label="loss")
        plt.plot(model_fit.history['val_loss'], "-", label="val_loss")
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(dir, name + '_loss_epochs_graph' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) +'.png'))

    @staticmethod
    def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        title = 'Confusion Matrix of {}'.format(title)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
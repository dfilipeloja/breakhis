import matplotlib.pyplot as plt
import os
from datetime import datetime


class Graph:

    @staticmethod
    def plot_accuracy_loss(model_fit, dir):
        plt.plot(model_fit.history['accuracy'], "-", label="accuracy")
        plt.plot(model_fit.history['val_accuracy'], "-", label="val_acc")
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(dir, 'accuracy_epochs_graph.png'))

        plt.plot(model_fit.history['loss'], "-", label="loss")
        plt.plot(model_fit.history['val_loss'], "-", label="val_loss")
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(dir, 'loss_epochs_graph' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) +'.png'))

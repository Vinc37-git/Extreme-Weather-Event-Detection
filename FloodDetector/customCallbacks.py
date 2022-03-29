'''
@brief:
A module for custom callbacks based on tensorflow callbacks.
'''

import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as tf_callbacks
from IPython.display import clear_output


class PlotTrainingProgress(tf_callbacks.Callback):
    
    def __init__(self, clear=True):
        '''Live plot of the training progress.
        Args:
            - clear: clear the output.
        '''
        self.clear = clear
    
    def on_train_begin(self, logs={}):
        '''Initialize all metrics on training begin.'''
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        
    def on_epoch_end(self, epoch, logs=None):
        '''Extract updates from logs, then clear output and plot training graph.'''
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        if self.clear:
            clear_output(wait=True)
            
        metrics = [metric for metric in logs if 'val' not in metric]
        
        self.fig, self.ax = plt.subplots(1, len(metrics), figsize=(15,5))

        for i, metric in enumerate(metrics):
            self.ax[i].plot(range(1, epoch + 2), self.metrics[metric], label=metric)
            
            if 'val_' + metric in logs:
                self.ax[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric)

            self.ax[i].legend()
            self.ax[i].grid()

        plt.tight_layout()
        plt.show()
        
    def on_train_end(self, logs=None):
        self.fig.savefig("plots/train_val_loss.png")   
        
        
class ShowSegmentationPrediction(tf_callbacks.Callback):
    '''Note: Currently, not working.'''
    
    def on_epoch_end(self, epoch, logs=None):
        ID = 1
        pred = self.model.predict([ds_test_Img_norm[ID]])
        plot_rand_imgs(1, ds_test_Img_norm[ID], ds_test_GT[ID], pred=pred, id_of_interest=[ID]);
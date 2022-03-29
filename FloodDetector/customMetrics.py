'''
@brief:
A module for custom metrics.
'''

import tensorflow.keras.metrics as tf_metrics
from tensorflow import argmax, newaxis


# from https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical
class SparseMeanIoU(tf_metrics.MeanIoU):
    '''Calculates the Mean Intersection over Union.
    There should be `# classes` floating point values per feature 
    for `y_pred` and a single floating point value per feature for `y_true`. '''
    
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
        super(SparseMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

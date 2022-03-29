'''
@brief
The Flood Detecot model based on the UNET architecture.
The Downsampler (encoder) is a pretrained model `MobileNetV2`, 
while the Upsampler (decoder) is a sequence of transposed Convolution, Normalization and Dropout Layers.

Additionally, a simple threshold based model is provided as baseline model.
'''

import numpy as np
import tensorflow as tf

#from tensorflow import argmax, newaxis
#from tensorflow.keras import Model
#from tensorflow.keras.models import load_model
#from tensorflow.keras.backend import clear_session
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Input, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Dropout
#from tensorflow.keras.applications import MobileNetV2, ResNet50
#from tensorflow.keras.applications.mobilenet import preprocess_input
#from tensorflow.keras.utils import plot_model

#import tensorflow.config as tfconf
#import tensorflow.keras.losses as losses
#import tensorflow.keras.callbacks as callbacks
#import tensorflow.keras.metrics as metrics
#import tensorflow.keras.optimizers as optim


def init_samplers(input_shape):
    ''' 
    Initialize Up- and Downsampler. 
    The Downsampler (encoder) is a pretrained model `MobileNetV2`, 
    while the Upsampler (decoder) is a sequence of transposed Convolution, Normalization and Dropout Layers.
    '''

    ### Initialize pretrained model ###
    pretrained_downsampler = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    ### prepare skip connectors
    layer_names = [
        'block_1_expand_relu',   # 256x256x96
        'block_3_expand_relu',   # 128x128x144
        'block_6_expand_relu',   # 64x64x192
        'block_13_expand_relu',  # 32x32x576
        'out_relu',              # 16x16x1280
    ]
    pretrained_downsampler_outputs = [pretrained_downsampler.get_layer(name).output for name in layer_names]


    ### compose encoders ###
    #encoder_top = Model(inputs=top_downsampler.input, outputs=top_downsampler.output)
    encoder = tf.keras.Model(name="MobileNetV2", inputs=pretrained_downsampler.input, outputs=pretrained_downsampler_outputs)

    encoder.trainable = False

    ### initialize decoder ###
    def upsample(name, filters, kernel_size=(3,3), padding='same', activation='relu'):
        # initializer = tf.random_normal_initializer(0., 0.02) for kernel_initializer AND maybre relu after normalization: https://www.tensorflow.org/tutorials/generative/pix2pix
        model = tf.keras.models.Sequential(name=name)
        model.add(tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        return model

    decoder = [
        upsample("up_1", 256),  #32x32x256 + Concat  # TODO: TAKE DOUBLE OF EVERY VALUE
        upsample("up_2", 128),  #64x64x128 + Concat
        upsample("up_3", 64),  #128x128x64
        upsample("up_4", 32),  #256x256x32
        upsample("up_5", 16) #, kernel_size=(1,1), activation=None),  # last layer is a 1x1 convolution
    ]
    
    return encoder, decoder

def create_encoder(input_shape, trainable=False):
    ''' Note: Function is currently not used.'''
    ### Initialize pretrained model ###
    pretrained_downsampler = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)

    ### prepare skip connectors
    layer_names = [
        'block_1_expand_relu',   # 256x256x96
        'block_3_expand_relu',   # 128x128x144
        'block_6_expand_relu',   # 64x64x192
        'block_13_expand_relu',  # 32x32x576
        'out_relu',              # 16x16x1280
    ]
    pretrained_downsampler_outputs = [pretrained_downsampler.get_layer(name).output for name in layer_names]

    ### compose encoders ###
    encoder = tf.keras.Model(name="MobileNetV2", inputs=pretrained_downsampler.input, outputs=pretrained_downsampler_outputs)

    encoder.trainable = trainable
    
    return encoder


def create_UNET(input_shape, name="model"):
    ''' 
    Create an instance of model based on the UNET architecture.
    The Downsampler (encoder) is a pretrained model `MobileNetV2`, 
    while the Upsampler (decoder) is a sequence of transposed Convolution, Normalization and Dropout Layers.
    '''
    
    tf.keras.backend.clear_session()
    
    encoder, decoder = init_samplers(input_shape)
    
    inputs = tf.keras.layers.Input(name="image_input", shape=input_shape)
    
    # encode the data. The encoder returns a list with all skip levels outputs.
    extracted_features = encoder(inputs)
    
    # The last element in the list keeps the last output at the bottleneck
    x = extracted_features[-1]
    
    # prepare skip connection establishing. Reverse skip levels (bottom-to-top) except of last one
    skips = reversed(extracted_features[:-1])
    
    # Upsampling and establish skip connections
    i = 1
    for up, skip in zip(decoder, skips):
        x = up(x)  # go one layer up
        x = tf.keras.layers.Concatenate(name=f"skip_con_{i}")([x, skip])  # establish skip connection
        i = i + 1
    
    x = decoder[-1](x)
    output = tf.keras.layers.Conv2D(name="out", filters=3, kernel_size=(1,1), padding='same', activation=None)(x),  # last layer is a 1x1 convolution
        
    return tf.keras.Model(name=name, inputs=inputs, outputs=output)


class ThresholdClassifier:

    def __init__(self, thresholds):
        ''' A simple threshold based classifier.
        Args:
            * thresholds: a list of thresholds. Note that only as many features as elements are given in this list are used to perform the classification.
            
        Note: Currently, the model is not general yet and is only working with Sen1Floods11 Data.
        '''
        self.ths = thresholds if type(thresholds) == list else list(thresholds)
        
    def predict(self, images):
        assert len(self.ths) == 2 and images.shape[-1] == 3
        
        '''The shifted Sen1Floods11 Data labeled the void pixels as 0, Non-Water as 1 and Water as 2.'''
        
        pred = np.zeros(images.shape, dtype='uint8')
        
        pred[(images[...,0] > self.ths[0]) & (images[...,0] > self.ths[1]), 1] = 1  # probabilites for label 1
        pred[(images[...,1] <= self.ths[0]) & (images[...,1] <= self.ths[1]), 2] = 1  # probabilites for label 2
        return pred
        
        







if __name__ == "__main__":
    pass




'''
def upsample(x, name, filters, kernel_size=(3,3), padding='same', activation='relu'):
    x = Conv2DTranspose(name=name, filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x


def create_decoder(x, skips):
        
    decoder_steps = [
        {"name":"up_1", "filters":256},  # TODO: TAKE DOUBLE OF EVERY VALUE
        {"name":"up_2", "filters":128},
        {"name":"up_3", "filters":64},
        {"name":"up_4", "filters":32},
        {"name":"out", "filters":3, "kernel_size":(1,1), "activation":None},  # last layer is a 1x1 convolution
    ]
    
    i = 1
    for up, skip in zip(decoder_steps, skips):
        x = upsample(x, **up)  # go one layer up
        x = Concatenate(name=f"skip_con_{i}")([x, skip])  # establish skip connection
        i = i + 1
    
    output = upsample(x, **decoder_steps[-1])
    
    return Model(name="Decoder", inputs=skips, outputs=output)
        
    
    
def create_UNET(input_shape):
    inputs = Input(name="image_input", shape=input_shape)
    
    encoder = create_encoder(input_shape)
    
    # encode the data. The encoder returns a list with all skip levels outputs.
    extracted_features = encoder(inputs)
    
    # The last element in the list keeps the last output at the bottleneck
    x = extracted_features[-1]
    
    # prepare skip connection establishing. Reverse skip levels (bottom-to-top) except of last one
    skips = reversed(extracted_features[:-1])
    
    decoder = create_decoder(x, skips)
    
    return Model(name="Flood_detector", inputs=inputs, outputs=decoder.output)
    
'''
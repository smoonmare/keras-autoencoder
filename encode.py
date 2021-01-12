import os
import re

from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(0)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTransponse, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf

# The encoder
img_input = Input(shape=(256, 256, 3))
# Creating 64 (3, 3) filters for 1 convolution layer
layer_1 = Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(img_input)
layer_2 = Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_1)
layer_3 = MaxPooling2D(padding='same')(layer_2)
layer_4 = Conv2D(128, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_3)
layer_5 = Conv2D(128, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_4)
layer_6 = MaxPooling2D(padding='same')(layer_5)
layer_7 = Conv2D(256, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_6)
encoder = Model(img_input, layer_7)
# encoder.summary() # summary of our model

# Decoder
layer_8 = UpSampling2D()(layer_7)
layer_9 = Conv2D(128, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_8)
layer_10 = Conv2D(128, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_9)
layer_11 = add([layer_5, layer_10])
layer_12 = UpSampling2D()(layer_11)
layer_13 = Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_12)
layer_14 = Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_13)
layer_15 = add([layer_2, layer_14])
decoded = Conv2D(3, (3, 3),
                 padding='same',
                 activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(layer_15)
autoencoder = Model(img_input, decoded)
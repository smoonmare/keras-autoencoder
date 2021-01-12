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
                 activity_reglarizer=regularizers.l1(10e-10))(layer_6)
encoder = Model(img_input, layer_7)
# encoder.summary() # summary of our model
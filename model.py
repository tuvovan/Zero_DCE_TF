import keras
import keras.backend as K

from keras import Sequential, Input, Model
from keras.layers import Conv2D, Activation, Concatenate
import math
import numpy as np

def DCE_x(input_shape):
    input_img = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(input_img)
    conv2 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv3)

    int_con1 = Concatenate(axis=-1)([conv4, conv3])
    conv5 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(int_con1)
    int_con2 = Concatenate(axis=-1)([conv5, conv2])
    conv6 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(int_con2)
    int_con3 = Concatenate(axis=-1)([conv6, conv1])
    x_r = Conv2D(24, (3,3), strides=(1,1), activation='tanh', padding='same')(int_con3)


    r1, r2, r3, r4, r5, r6, r7, r8 = x_r[:,:,:,:3], x_r[:,:,:,3:6], x_r[:,:,:,6:9], x_r[:,:,:,9:12], x_r[:,:,:,12:15], x_r[:,:,:,15:18], x_r[:,:,:,18:21], x_r[:,:,:,21:24]
    x = input_img + r1 * (K.pow(input_img,2)-input_img)
    x = x + r2 * (K.pow(x,2)-x)
    x = x + r3 * (K.pow(x,2)-x)
    enhanced_image_1 = x + r4*(K.pow(x,2)-x)
    x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
    x = x + r6*(K.pow(x,2)-x)	
    x = x + r7*(K.pow(x,2)-x)
    enhance_image = x + r8*(K.pow(x,2)-x)

    model = Model(inputs=input_img, outputs = [enhanced_image_1, enhance_image, x_r])

    return model

class DCE(Model):
    def __init__(self):
        super(DCE, self).__init__()
        self.conv = Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same')
        self.conv_32 = Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same')
        self.conv_64 = Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same')
        self.cat = Concatenate(axis=-1)
        self.conv_last = Conv2D(24, (3,3), strides=(1,1), activation='tanh', padding='same')
        #self.load_weights()
    def call(self, input_img):
        conv1 = self.conv(input_img)
        conv2 = self.conv_32(conv1)
        conv3 = self.conv_32(conv2)
        conv4 = self.conv_32(conv3)

        int_con1 = self.cat([conv4, conv3])
        conv5 = self.conv_64(int_con1)
        int_con2 = self.cat([conv5, conv2])
        conv6 = self.conv_64(int_con2)
        int_con3 = self.cat([conv6, conv1])
        x_r = self.conv_last(int_con3)


        r1, r2, r3, r4, r5, r6, r7, r8 = x_r[:,:,:,:3], x_r[:,:,:,3:6], x_r[:,:,:,6:9], x_r[:,:,:,9:12], x_r[:,:,:,12:15], x_r[:,:,:,15:18], x_r[:,:,:,18:21], x_r[:,:,:,21:24]
        x = input_img + r1 * (K.pow(input_img,2)-input_img)
        x = x + r2 * (K.pow(x,2)-x)
        x = x + r3 * (K.pow(x,2)-x)
        enhanced_image_1 = x + r4*(K.pow(x,2)-x)
        x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(K.pow(x,2)-x)	
        x = x + r7*(K.pow(x,2)-x)
        enhance_image = x + r8*(K.pow(x,2)-x)

        return enhanced_image_1, enhance_image, x_r

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)


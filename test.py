import os
import sys
import glob
import time
import argparse 
import numpy as np

import tensorflow as tf 
import tensorflow.keras.backend as K

from PIL import Image
from src.model import DCE_x
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D

tf.compat.v1.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--lowlight_test_images_path', type=str, default="test/LIME/")
config = parser.parse_args()

def test(lowlight_test_images_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    input_img = Input(shape=(512, 512, 3))
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

    model = Model(inputs=input_img, outputs = x_r)
    model.load_weights("weights/best.h5")

    ### load image ###
    for test_file in glob.glob(lowlight_test_images_path + "*.bmp"):
        data_lowlight_path = test_file
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])

        original_img = original_img.resize((512,512), Image.ANTIALIAS) 
        original_img = (np.asarray(original_img)/255.0)

        img_lowlight = Image.open(data_lowlight_path)
                
        img_lowlight = img_lowlight.resize((512,512), Image.ANTIALIAS)

        img_lowlight = (np.asarray(img_lowlight)/255.0) 
        img_lowlight = np.expand_dims(img_lowlight, 0)
        # img_lowlight = K.constant(img_lowlight)

        ### process image ###
        A = model.predict(img_lowlight)
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
        x = original_img + r1 * (K.pow(original_img,2)-original_img)
        x = x + r2 * (K.pow(x,2)-x)
        x = x + r3 * (K.pow(x,2)-x)
        enhanced_image_1 = x + r4*(K.pow(x,2)-x)
        x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(K.pow(x,2)-x)	
        x = x + r7*(K.pow(x,2)-x)
        enhance_image = x + r8*(K.pow(x,2)-x)
        enhance_image = tf.cast((enhance_image[0,:,:,:] * 255), dtype=np.uint8)
        enhance_image = Image.fromarray(enhance_image.numpy())
        enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)
        enhance_image.save(test_file.replace(".bmp", "_rs.bmp"))

test(config.lowlight_test_images_path)

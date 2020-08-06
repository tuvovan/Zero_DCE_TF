import keras.backend as K 
import keras.applications.vgg16 as vgg16
import math
import numpy as np
import tensorflow as tf 

from tensorflow.keras.layers import Conv2D, AveragePooling2D


def L_color(x):
    mean_rgb = tf.reduce_mean(x,[1,2],keepdims=True)
    mr,mg, mb = mean_rgb[:,:,:,0], mean_rgb[:,:,:,1], mean_rgb[:,:,:,2] 
    Drg = tf.pow(mr-mg,2)
    Drb = tf.pow(mr-mb,2)
    Dgb = tf.pow(mb-mg,2)
    l = tf.pow(K.pow(Drg,2) + tf.pow(Drb,2) + tf.pow(Dgb,2),0.5)

    return l


def L_exp(x, mean_val=0.6):
    x = tf.reduce_mean(x,3,keepdims=True)
    mean = AveragePooling2D(pool_size=16, strides=16)(x) # non overlap

    d = tf.reduce_mean(tf.pow(mean- mean_val,2))
    return d
        
def L_TV(x, TVLoss_weight=1):
    batch_size = tf.shape(x)[0].numpy()
    h_x = tf.shape(x)[1].numpy()
    w_x = tf.shape(x)[2].numpy()
    count_h =  (tf.shape(x)[2].numpy()-1) * tf.shape(x)[3].numpy()
    count_w = tf.shape(x)[2].numpy() * (tf.shape(x)[3].numpy() - 1)
    h_tv = tf.reduce_sum(tf.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2))
    w_tv = tf.reduce_sum(tf.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2))
    return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

def init_f_l(shape, dtype=None):
    ker = np.zeros(shape, dtype=None)
    ker[1][0] = -1
    ker[1][1] = 1
    return ker

def init_f_r(shape, dtype=None):
    ker = np.zeros(shape, dtype=None)
    ker[1][2] = -1
    ker[1][1] = 1
    return ker

def init_f_u(shape, dtype=None):
    ker = np.zeros(shape, dtype=None)
    ker[0][1] = -1
    ker[1][1] = 1
    return ker

def init_f_d(shape, dtype=None):
    ker = np.zeros(shape, dtype=None)
    ker[2][1] = -1
    ker[1][1] = 1
    return ker

def L_spa(org, enhance):

    org_mean = tf.reduce_mean(org, 3, keepdims=True)
    enhance_mean = tf.reduce_mean(enhance,3,keepdims=True)

    org_pool =  AveragePooling2D(pool_size=4)(org_mean)		
    enhance_pool = AveragePooling2D(pool_size=4)(enhance_mean)

    D_org_left = Conv2D(kernel_initializer=init_f_l, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_right = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_up = Conv2D(kernel_initializer=init_f_u, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_down = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(org_pool)

    D_enhance_letf = Conv2D(kernel_initializer=init_f_l, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_right = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_up = Conv2D(kernel_initializer=init_f_u, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_down = Conv2D(kernel_initializer=init_f_d, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)

    D_left = tf.pow(D_org_left - D_enhance_letf,2)
    D_right = tf.pow(D_org_right - D_enhance_right,2)
    D_up = tf.pow(D_org_up - D_enhance_up,2)
    D_down = tf.pow(D_org_down - D_enhance_down,2)
    E = (D_left + D_right + D_up +D_down)

    return E

import keras
import keras.backend as K 
import keras.applications.vgg16 as vgg16
import math
import numpy as np

from keras.layers import Conv2D, AveragePooling2D


def L_color(x):
    b,h,w,c = K.int_shape(x)[0], K.int_shape(x)[1], K.int_shape(x)[2], K.int_shape(x)[3]

    mean_rgb = K.mean(x,[1,2],keepdims=True)
    mr,mg, mb = mean_rgb[:,:,:,0], mean_rgb[:,:,:,1], mean_rgb[:,:,:,2] 
    Drg = K.pow(mr-mg,2)
    Drb = K.pow(mr-mb,2)
    Dgb = K.pow(mb-mg,2)
    l = K.pow(K.pow(Drg,2) + K.pow(Drb,2) + K.pow(Dgb,2),0.5)

    return l


def L_exp(x, mean_val=0.6):
    b,h,w,c = K.int_shape(x)[0], K.int_shape(x)[1], K.int_shape(x)[2], K.int_shape(x)[3]
    x = K.mean(x,3,keepdims=True)
    mean = AveragePooling2D(pool_size=16, strides=16)(x) # non overlap

    d = K.mean(K.pow(mean- mean_val,2))
    return d
        
def L_TV(x, TVLoss_weight=1):
    batch_size = K.int_shape(x)[0]
    h_x = K.int_shape(x)[1]
    w_x = K.int_shape(x)[2]
    count_h =  (K.int_shape(x)[2]-1) * K.int_shape(x)[3]
    count_w = K.int_shape(x)[2] * (K.int_shape(x)[3] - 1)
    h_tv = K.sum(K.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2))
    w_tv = K.sum(K.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2))
    return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

def init_f_l(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[1][0] = -1
    ker[1][1] = 1
    return ker

def init_f_r(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[1][2] = -1
    ker[1][1] = 1
    return ker

def init_f_u(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[0][1] = -1
    ker[1][1] = 1
    return ker

def init_f_d(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[2][1] = -1
    ker[1][1] = 1
    return ker

def L_spa(org, enhance):
    # kernel_left = K.variable(np.array([[0,0,0], [-1,1,0],[0,0,0]]))
    # kernel_right = K.variable(np.array([[0,0,0], [0,1,-1],[0,0,0]]))
    # kernel_up = K.variable(np.array([[0,-1,0], [0,1,0],[0,0,0]]))
    # kernel_down = K.variable(np.array([[0,0,0], [0,1,0],[0,-1,0]]))
    # kernel_left = init_f_l((3,3), string='l')
    # kernel_right = init_f_l((3,3), string='r')
    # kernel_up = init_f_l((3,3), string='u')
    # kernel_down = init_f_l((3,3), string='d')


    b,h,w,c = K.int_shape(org)[0], K.int_shape(org)[1], K.int_shape(org)[2], K.int_shape(org)[3]

    org_mean = K.mean(org, 3, keepdims=True)
    enhance_mean = K.mean(enhance,3,keepdims=True)

    org_pool =  AveragePooling2D(pool_size=4)(org_mean)		
    enhance_pool = AveragePooling2D(pool_size=4)(enhance_mean)

    D_org_left = Conv2D(kernel_initializer=init_f_l, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_right = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_up = Conv2D(kernel_initializer=init_f_u, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    D_org_down = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(org_pool)
    # D_org_letf = Conv2D(org_pool , kernel_left, padding=1)
    # D_org_right = Conv2D(org_pool , kernel_right, padding=1)
    # D_org_up = Conv2D(org_pool , kernel_up, padding=1)
    # D_org_down = Conv2D(org_pool , kernel_down, padding=1)

    D_enhance_letf = Conv2D(kernel_initializer=init_f_l, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_right = Conv2D(kernel_initializer=init_f_r, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_up = Conv2D(kernel_initializer=init_f_u, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)
    D_enhance_down = Conv2D(kernel_initializer=init_f_d, padding='same', kernel_size=(3,3), filters=1)(enhance_pool)

    D_left = K.pow(D_org_left - D_enhance_letf,2)
    D_right = K.pow(D_org_right - D_enhance_right,2)
    D_up = K.pow(D_org_up - D_enhance_up,2)
    D_down = K.pow(D_org_down - D_enhance_down,2)
    E = (D_left + D_right + D_up +D_down)

    return E

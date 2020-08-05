import keras
import tensorflow as tf 
import keras.backend as K
import os
import sys
import argparse 
import time
import data_lowlight
import model
import numpy as np

from loss import *
from model import DCE_x
from keras import Model, Input
from keras.layers import Concatenate, Conv2D


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_dataset = data_lowlight.DataGenerator(config.lowlight_images_path, config.train_batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

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
    

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                A = model(img_lowlight)
                r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
                x = input_img + r1 * (K.pow(input_img,2)-input_img)
                x = x + r2 * (K.pow(x,2)-x)
                x = x + r3 * (K.pow(x,2)-x)
                enhanced_image_1 = x + r4*(K.pow(x,2)-x)
                x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
                x = x + r6*(K.pow(x,2)-x)	
                x = x + r7*(K.pow(x,2)-x)
                enhance_image = x + r8*(K.pow(x,2)-x)
                
                loss_TV = 200*L_TV(A)
                loss_spa = K.mean(L_spa(enhance_image, img_lowlight))
                loss_col = 5*K.mean(L_color(enhance_image))
                loss_exp = 10*K.mean(L_exp(enhance_image, mean_val=0.6))

                total_loss = loss_TV + loss_spa + loss_col + loss_exp

            grads = tape.gradient(total_loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if iteration % config.snapshot_iter == 0:
                model.save_weights(os.path.join(config.snapshots_folder, "Epoch"+str(epoch)+'.h5'))




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="G:\\My Drive\\Dataset_Part1\\All\\")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="weights\\")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "weights\\Epoch99.h5")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
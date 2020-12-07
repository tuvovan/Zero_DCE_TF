import os
import sys
import glob
import argparse 
import time
import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K

from src import *
from tensorflow import keras
from src import data_lowlight
from src.loss import *
from src.model import DCE_x
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D
from PIL import Image


def progress(epoch, trained_sample ,total_sample, bar_length=25, total_loss=0, message=""):
    percent = float(trained_sample) / total_sample
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rEpoch {0}: [{1}] {2}%  ----- Loss: {3}".format(epoch, hashes + spaces, int(round(percent * 100)), float(total_loss)) + message)
    sys.stdout.flush()

def eval(model):
    for data_lowlight_path in glob.glob("test/" + "*.jpg"):
        # load image
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])

        original_img = original_img.resize((512,512), Image.ANTIALIAS) 
        original_img = (np.asarray(original_img)/255.0)

        img_lowlight = Image.open(data_lowlight_path)
                    
        img_lowlight = img_lowlight.resize((512,512), Image.ANTIALIAS)

        img_lowlight = (np.asarray(img_lowlight)/255.0) 
        img_lowlight = np.expand_dims(img_lowlight, 0)
        # predict
        A = model.predict(img_lowlight) 
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
        x = original_img + r1 * (K.pow(original_img,2)-original_img)
        x = x + r2 * (tf.pow(x,2)-x)
        x = x + r3 * (tf.pow(x,2)-x)
        enhanced_image_1 = x + r4*(K.pow(x,2)-x)
        x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(tf.pow(x,2)-x)	
        x = x + r7*(tf.pow(x,2)-x)
        enhance_image = x + r8*(K.pow(x,2)-x)
        enhance_image = tf.cast((enhance_image[0,:,:,:] * 255), dtype=np.uint8)
        enhance_image = Image.fromarray(enhance_image.numpy())
        enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)
        enhance_image.save(data_lowlight_path.replace('.jpg', '_rs.jpg'))

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

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
    
    min_loss = 10000.0
    print("Start training ...")
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                A = model(img_lowlight)
                r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
                x = img_lowlight + r1 * (tf.pow(img_lowlight,2)-img_lowlight)
                x = x + r2 * (tf.pow(x,2)-x)
                x = x + r3 * (tf.pow(x,2)-x)
                enhanced_image_1 = x + r4*(tf.pow(x,2)-x)
                x = enhanced_image_1 + r5*(tf.pow(enhanced_image_1,2)-enhanced_image_1)		
                x = x + r6*(tf.pow(x,2)-x)	
                x = x + r7*(tf.pow(x,2)-x)
                enhance_image = x + r8*(tf.pow(x,2)-x)
                
                loss_TV = 200*L_TV(A)
                loss_spa = tf.reduce_mean(L_spa(enhance_image, img_lowlight))
                loss_col = 5*tf.reduce_mean(L_color(enhance_image))
                loss_exp = 10*tf.reduce_mean(L_exp(enhance_image, mean_val=0.6))

                total_loss = loss_TV + loss_spa + loss_col + loss_exp

            grads = tape.gradient(total_loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # if iteration % config.display_iter == 0:
            #     print("Training loss (for one batch) at step %d: %.4f" % (iteration, float(total_loss)))

            progress(epoch+1, (iteration+1), len(train_dataset), total_loss=total_loss)

            if (iteration+1) % config.checkpoint_iter == 0 and total_loss < min_loss:
                min_loss = total_loss
                progress(epoch+1, (iteration+1), len(train_dataset), total_loss=total_loss, message=' ----- saved weight for epoch ' + str(epoch+1) + ' iter ' + str(iteration+1))
                model.save_weights(os.path.join(config.checkpoints_folder, "ep_"+str(epoch+1)+"_it_"+str(iteration+1)+".h5"))




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="Dataset_Part1/All/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=2)
	parser.add_argument('--checkpoint_iter', type=int, default=2)
	parser.add_argument('--checkpoints_folder', type=str, default="weights/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "weights/Epoch10.h5")

	config = parser.parse_args()

	if not os.path.exists(config.checkpoints_folder):
		os.mkdir(config.checkpoints_folder)


	train(config)

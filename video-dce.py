import os
import argparse
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
from src.model import DCE_x
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D
import tensorflow.keras.backend as K

# Set up TensorFlow configuration for better GPU performance
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Setting up argument parser
parser = argparse.ArgumentParser(description="Enhance video quality using an image enhancement model.")
parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
parser.add_argument('--output_video', type=str, default="output.mkv", required=True, help='Path to the output video file.')
parser.add_argument('--max_frames', type=int, default=None, help="Maximum number of frames to process from the input video.")
parser.add_argument('--dar', type=str, default="4:3", help="Desired Display Aspect Ratio for the output video (e.g., '4:3', '16:9').")
args = parser.parse_args()

# Build the image enhancement model
input_img = Input(shape=(None, None, 3))
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

def enhance_frame(original_frame):
    original_frame_resized = np.asarray(Image.fromarray(original_frame).resize((512, 512), Image.LANCZOS)) / 255.0
    input_data = np.expand_dims(original_frame_resized, 0)
    
    A = model.predict(input_data)
    r1, r2, r3, r4, r5, r6, r7, r8 = [A[:, :, :, i:i+3] for i in range(0, 24, 3)]

    x = original_frame_resized + r1 * (K.pow(original_frame_resized, 2) - original_frame_resized)
    x = x + r2 * (K.pow(x, 2) - x)
    x = x + r3 * (K.pow(x, 2) - x)
    enhanced_image_1 = x + r4 * (K.pow(x, 2) - x)
    x = enhanced_image_1 + r5 * (K.pow(enhanced_image_1, 2) - enhanced_image_1)
    x = x + r6 * (K.pow(x, 2) - x)
    x = x + r7 * (K.pow(x, 2) - x)
    enhance_image = x + r8 * (K.pow(x, 2) - x)
    enhance_image = (enhance_image[0].numpy() * 255).astype(np.uint8)
    
    return enhance_image

def save_frames_as_video(input_video_path, output_video_path, max_frames, dar):
    # Load the input video using MoviePy
    video = VideoFileClip(input_video_path)
    fps = video.fps
    size = video.size

    # If max_frames is set, shorten the video clip accordingly
    if max_frames is not None:
        video = video.subclip(0, max_frames / fps)  # subclip takes start time and end time

    # Process the video frames
    enhanced_clip = video.fl_image(enhance_frame)

    # Set audio from the original video
    enhanced_clip = enhanced_clip.set_audio(video.audio)

    # Write the video with desired codec
    enhanced_clip.write_videofile(output_video_path, codec='ffv1', audio=True, ffmpeg_params=["-acodec", "copy", "-aspect", dar])

def main():
    save_frames_as_video(args.input_video, args.output_video, args.max_frames, args.dar)

if __name__ == "__main__":
    main()

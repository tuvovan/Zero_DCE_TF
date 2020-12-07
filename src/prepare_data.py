import glob
import shutil
import os 
import cv2

ori_path = 'Dataset_Part1/'
des_path = 'Dataset_Part1/All/'

def move_file(ori_path, des_path):
    dir_list = [x[0] for x in os.walk(ori_path)]
    for p in dir_list[1:]:
        for f in glob.glob(p + "/*.JPG"):
            name = f.split("/")[-2] + "_" + f.split("/")[-1]
            shutil.move(f, os.path.join(des_path, name))

move_file(ori_path, des_path)
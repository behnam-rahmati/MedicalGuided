from train_sunnybrook import train, map_all_contours, export_all_contours, get_elements
import os, fnmatch, sys
import random
import time
import os
import cv2
import numpy as np
from fcn_model import fcn_model
from chan_vese_1 import chanvese
import matplotlib.pyplot as plt
from skimage import exposure, measure
import tensorflow as tf
print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
import torch
# Check if CUDA is available
from keras import backend as K

SUNNYBROOK_ROOT_PATH = "C:\\Users\\user\\data"
TRAIN_CONTOUR_PATH = "C:\\Users\\Lab\\Desktop\\backup\\backup"
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
VAL_CONTOUR_PATH = "C:\\Users\\user\\data\\val_GT"
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\user\\data\\Sunnybrook Cardiac MR Database ContoursPart1\\Sunnybrook Cardiac MR Database ContoursPart1\\OnlineDataContours"
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online')
TEST_CONTOUR_PATH = "C:\\Users\\Lab\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_test')
                        


contour_type = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

crop_size = 100
input_shape = (crop_size, crop_size, 1)
num_classes = 2
test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
print(len(test_ctrs))
print("hhh")
img_dev, mask_dev = export_all_contours(test_ctrs,
                                        TEST_IMG_PATH,
                                        crop_size=crop_size)
                                        
save_folder1 = "C:\\Users\\Lab\\Desktop\\send\\newest run\\suponly\\52\\weights\\2"  
weight1 = os.path.join(save_folder1,'model.h5')    
model1 = fcn_model(input_shape, num_classes, weights=weight1)  
pred_masks1 = model1.predict(img_dev, batch_size=32, verbose=1) 

save_folder2 = "C:\\Users\\Lab\\Desktop\\send\\newest run\\DM\\0.7\\weights\\2"
weight2 = os.path.join(save_folder2,'model.h5')    
model2 = fcn_model(input_shape, num_classes, weights=weight2)  
pred_masks2 = model2.predict(img_dev, batch_size=32, verbose=1) 

print(pred_masks1.shape)
print(pred_masks2.shape)

for i in range(len(test_ctrs)):
    img = img_dev[i,:,:,0]
    gt_mask = mask_dev[i,:,:,0]
    pred_mask1 = pred_masks1[i,:,:,0]
    pred_mask2 = pred_masks2[i,:,:,0]
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray')
    
    # Overlay ground truth contour in green
    contours = measure.find_contours(gt_mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=4, color='green')
    
    # Overlay first set of predictions in red
    contours = measure.find_contours(pred_mask1, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color='red')
    
    # Overlay second set of predictions in yellow
    contours = measure.find_contours(pred_mask2, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color='yellow')
    
    plt.show()
    # Save figure
    save_path = os.path.join('predictions', 'image_{}.png'.format(i))
    # plt.savefig(save_path, bbox_inches='tight')
    
    # Close figure to save memory
    plt.close(fig)
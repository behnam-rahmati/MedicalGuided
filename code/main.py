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
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("Code is running on GPU")
else:
    print("Code is running on CPU")
seed = 1234
np.random.seed(seed)

## Enter the dataset paths here
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

## initial variables (contour_type can be either 'i' for endocardium or 'o' for epicardium)
contour_type = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

crop_size = 100
input_shape = (crop_size, crop_size, 1)
num_classes = 2                        

## This function saves the images with segementation contours extracted from the predicted masks                        
def save_images_with_contours(imgs, masks, gt, output_dir):
    """
    Save images with contours drawn on the segmentation masks as JPEG images.
    Args:
        imgs: NumPy array of shape (num_imgs, height, width, 1) representing the original images.
        masks: NumPy array of shape (num_imgs, height, width, 1) representing the segmentation masks.
        gt: NumPy array of shape (num_imgs, height, width, 1) representing the ground truth segmentation.
        output_dir: String representing the directory where the output images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(imgs.shape[0]):
        img = imgs[i].squeeze()
        # Normalize the image to [0, 1] range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 35  # Brightness control (0-100)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)      
        
        mask = masks[i].squeeze()
        gt_mask = gt[i].squeeze()
        mask = np.where(mask > 0.5, 255, 0).astype('uint8')
        img = img.astype('uint8')
        gt_mask = gt_mask.astype('uint8')
        
        # Create a mask for the ground truth and predicted contours
        gt_contour = cv2.findContours(gt_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        pred_contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        
        # Create an RGB image to draw the contours on
        img_rgb = np.stack((img,)*3, axis=-1).astype('uint8')

        # Draw the ground truth and predicted contours on the RGB image
        img_rgb = cv2.drawContours(img_rgb, gt_contour, -1, (0, 255, 0), 1)
        img_rgb = cv2.drawContours(img_rgb, pred_contour, -1, (0, 0, 255), 1)
        img_rgb = cv2.resize(img_rgb, (400, 400), interpolation=cv2.INTER_AREA)
        
        # Save the result to a JPEG file
        filename = os.path.join(output_dir, f"segmentation_result_{i}.jpg")
        plt.imsave(filename, img_rgb)
        

## This function saves the results of the experiments in a text file     
def save_results(folder_name, file_name, data):
    # Get the current directory where the Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a directory to save the file in
    directory = folder_name
    if not os.path.exists(os.path.join(current_dir, directory)):
        os.makedirs(os.path.join(current_dir, directory))

    # List of numbers to save to a file
 
    # Define the filename and path of the file to save
    filename = file_name
    filepath = os.path.join(current_dir, directory, filename)

    # Open the file in write mode and save the list of numbers
    with open(filepath, "w") as f:
        for number in data:
            f.write(str(number) + "\n")
            
    # Confirm that the file has been saved by printing its contents
    with open(filepath, "r") as f:
        file_contents = f.read()

        
## test contours are selected only once for all the tasks to have a fair comparison     
test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
indices_test = [random.randint(0, len(test_ctrs)-1) for i in range(50)]
save_results('test_ctrs', 'indices.txt',indices_test)


test_ctrs_selected = get_elements(test_ctrs,indices_test)
train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))

img_all, mask_all = export_all_contours(train_ctrs,
                                            TRAIN_IMG_PATH,
                                            crop_size=crop_size)  
                                       
img_dev, mask_dev = export_all_contours(test_ctrs_selected,
                                        TEST_IMG_PATH,
                                        crop_size=crop_size)

    
for itr in range (0,5):

########################################################################################
# # # # # # # # # # # preparing the labeled and unlabeled train data # # # # # # # # # #

    num_ctrs = len(img_all)/20*itr + len(img_all)/10 #create randomly 10-15-20-25 percent of the train data
    indices_selected = []
    train_indices = []
    indices_selected = random.sample(range(len(train_ctrs)-1), num_ctrs)
    

    not_indices = [i2 for i2 in range(len(train_ctrs)) if i2 not in indices_selected]
    unlabeled_indices = not_indices
    
    train_indices = indices_selected
 
    train_ctrs_selected = get_elements(train_ctrs,train_indices)
    
    img_train, mask_train = export_all_contours(train_ctrs_selected,
                                        TRAIN_IMG_PATH,
                                        crop_size=crop_size) 
                                   
    ######################################################################################                                    
    # # # # # # # # # # # # # # # # supervised only  # # # # # # # # # # # # # # # # # # #                                   
                                        
    save_folder = 'suponly\\'+ str(num_ctrs)+'\\weights\\'+str(itr)
    restmp, epoch_tmp = (train(img_train, mask_train, img_dev, mask_dev, save_folder))
    save_results ('suponly\\'+ str(num_ctrs)+'\\train_ctrs','ctrs'+str(itr)+'.txt', indices_selected)    
    save_results ('suponly\\'+ str(num_ctrs)+'\\results','indices'+str(itr)+'.txt', [restmp])       
    suponly_weights = os.path.join(save_folder,'model.h5')    
    model1 = fcn_model(input_shape, num_classes, weights=suponly_weights)  
    pred_masks_all = model1.predict(img_all, batch_size=32, verbose=1) 
    
    ## create the pseudo-labels based on the initially trained network    
    pred_masks_all[pred_masks_all>=0.5] = 1
    pred_masks_all[pred_masks_all<0.5] = 0
    pred_masks_dev = model1.predict(img_dev, batch_size=32, verbose=1) 
    pred_masks_dev[pred_masks_dev>=0.5] = 1
    pred_masks_dev[pred_masks_dev<0.5] = 0
    save_images_with_contours(img_dev, pred_masks_dev, mask_dev, output_dir = "suponly")
    
    ########################################################################################
    # # # ## # # # # # # # # # # # # apply pseudo-labeling # # # # # # # # # # # # # # # # # 
    
    mask_all[unlabeled_indices] = pred_masks_all [unlabeled_indices]
    
    
    save_folder = 'PL\\'+ str(num_ctrs)+'\\nofilter\\weights\\'+str(itr)
    PLtmp,PLepoch_tmp = (train(img_all, mask_all, img_dev, mask_dev, save_folder))
    save_results ('PL\\'+ str(num_ctrs)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [PLtmp])   

    weights = os.path.join(save_folder,'model.h5')    
    model1 = fcn_model(input_shape, num_classes, weights=weights)  
    pred_masks_dev = model1.predict(img_dev, batch_size=32, verbose=1) 
    pred_masks_dev[pred_masks_dev>=0.5] = 1
    pred_masks_dev[pred_masks_dev<0.5] = 0
    save_images_with_contours(img_dev, pred_masks_dev, mask_dev, output_dir = "pseudo-labeling")    
            

    
    ######################################################################################
    # # # # # # # # # #  apply deformable models + total variation # # # # # # # # # # # # 
 
    shape_weights = [0.5, 0.4,  0.6, 0.7, 0.8, 0.95]
    TV_weights = [0, 0.0001, 0.0002]
    dice_results = []
    best_dice = 0

    for shape_weight in shape_weights:  
        for TV_weight in TV_weights:
            masks_deformable = mask_all  

            save_folder = 'DM\\'+ str(shape_weight)+str(TV_weight)+'\\weights\\'+str(itr)        
            for index in unlabeled_indices:
                img = img_all[index]
                img = img[:,:,0]
                mask = pred_masks_all[index]
                mask = mask[:,:,0]    
                mask[mask<0.5] = 0
                mask [ mask>=0.5] = 1
                seg, _, _ = chanvese(I = img, init_mask = mask, max_its=200, display=False, alpha=0.5 , shape_w = shape_weight)    
                masks_deformable[index] = np.expand_dims(seg, axis=2)         
                
            deftmp,defepoch_tmp = (train(img_all, masks_deformable, img_dev, mask_dev, save_folder, TV_param = TV_weight))
            dice_results.append(deftmp[0])
            if(deftmp[0]>best_dice):
                best_dice = deftmp[0]
            save_results ('DM\\'+ str(num_ctrs)+str(TV_weight)+'\\results','indices'+str(shape_weight)+str(TV_weights)+'.txt', [deftmp])  

            weights = os.path.join(save_folder,'model.h5')    
            model1 = fcn_model(input_shape, num_classes, weights=weights)  
            pred_masks_dev = model1.predict(img_dev, batch_size=32, verbose=1) 
            pred_masks_dev[pred_masks_dev>=0.5] = 1
            pred_masks_dev[pred_masks_dev<0.5] = 0
            save_images_with_contours(img_dev, pred_masks_dev, mask_dev, output_dir = 'deformable_models' + str(shape_weight)+str(TV_weight))  

    max_index = dice_results.index(max(dice_results))
        
    ######################################################################################################
    # # # ## # # # # # # # # # # # # Only total variation  # # # # # # # # # # # # # # # # # # # # # # # # 
    TV_weights = [0, 0.0001, 0.00005]   
    for TV_weight in TV_weights:
    
    
        mask_all[unlabeled_indices] = pred_masks_all [unlabeled_indices]
        
        
        save_folder = 'PL\\'+ str(num_ctrs)+str(TV_weight)+'\\nofilter\\weights\\'+str(itr)
        PLtmp,PLepoch_tmp = (train(img_all, mask_all, img_dev, mask_dev, save_folder, TV_param = TV_weight))
        save_results ('PL\\'+ str(num_ctrs)+str(TV_weight)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [PLtmp])     

        
    
       
    
                  
        

    
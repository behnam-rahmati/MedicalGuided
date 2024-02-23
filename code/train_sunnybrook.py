#!/usr/bin/env python2.7

import pydicom, cv2, re, math, shutil
import os, fnmatch, sys
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from itertools import izip
from fcn_model import *
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
# from U_net import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler


print(tf.__version__)
seed = 1234
np.random.seed(seed)
#SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = "C:\\Users\\user\\data"
TRAIN_CONTOUR_PATH = "C:\\Users\\user\\OneDrive\\Desktop\\backup\\backup"
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
VAL_CONTOUR_PATH = "C:\\Users\\user\\data\\val_GT"
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\user\\data\\Sunnybrook Cardiac MR Database ContoursPart1\\Sunnybrook Cardiac MR Database ContoursPart1\\OnlineDataContours"
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online')
TEST_CONTOUR_PATH = "C:\\Users\\user\\OneDrive\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_test')
ONLY_TRAIN_CONTOUR_PATH = "C:\\Users\\user\\OneDrive\\Desktop\\backup\\30\\only train"

# class UpdateIgnoreMatrix(Callback):
    # def __init__(self, ignore_matrix):
        # super().__init__()
        # self.ignore_matrix = ignore_matrix
    # def on_batch_begin(self, batch, logs=None):
        # # update ignore matrix for current batch
        # self.model.loss_functions[0].ignore_matrix = self.ignore_matrix[batch]
        



def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


    
def save_results(folder_name, file_name, data):
    # Get the current directory where the Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
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
        print(file_contents)
        
def get_elements(list, indices):
    elements = []
    for i in indices:
        elements.append(list[i])
    return elements
    
def dice_coef(y_true, y_pred, smooth=0.0):
    # print(np.array(y_true).shape)
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
    
def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

    
class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))
        self.slice_no =  math.floor(self.img_no/20) if self.img_no%20 !=0 else math.floor(self.img_no/20)-1
        self.ED_flag = True if ((self.img_no%20) < 10 and (self.img_no % 20) !=0) else False
        self.is_weak = 0
   
    
    def __str__(self):
        return 'Contour for case %s, image %d' % (self.case, self.img_no)
    
    __repr__ = __str__
def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8') # shape is 256, 256


    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    # print(coords.shape)   (num_points , 2)
    # print("this is coords shape")
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask

def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    #for dirpath, dirnames, files in os.walk(contour_path):
    #    print(files)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
        
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours
    
def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(list(contours))))
    print(len(contours))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask        
    return images, masks  
    
def train(img_train, mask_train, img_dev, mask_dev, save_folder, weights_path= None, pixel_filtering = False , thresh=0.5, TV_param = 0):  
    model_checkpoint_callback = ModelCheckpoint(
        filepath= save_folder + '\\model.h5',
        save_weights_only=False,
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True, verbose=0)
    tbCallBack = TensorBoard(log_dir='./Graph4', histogram_freq=0, write_graph=True, write_images=True)  
    crop_size = 100
    contour_type = 'i'
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    #weights = 'C:\\Users\\user\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=weights_path)
    epochs = 40
    # mini_batch_size = 1
    reliable_pixels = np.logical_or(mask_train < 1-thresh , mask_train >= thresh) 
    ignore_matrix = np.zeros_like(mask_train)
    if(pixel_filtering == True):
        ignore_matrix [reliable_pixels ==False] = 1
    mask_train[mask_train < 0.5] = 0
    mask_train[mask_train >= 0.5] = 1
    mask_train_custom = np.concatenate((mask_train, ignore_matrix), axis=3)
    mask_dev = np.concatenate((mask_dev, np.zeros_like(mask_dev)), axis=3)
    print(mask_dev.shape ,"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    lrate = LearningRateScheduler(step_decay, verbose=0)

    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=masked_loss(TV_param),
           metrics=['accuracy', dice_coef, jaccard_coef, hausdorff_distance], run_eagerly = True)      
    history = model.fit(img_train, mask_train_custom, batch_size=1, epochs=40, verbose=0, validation_data=(img_dev,mask_dev),
          callbacks=[model_checkpoint_callback, tbCallBack])
    print(history.history.keys())
    dice = history.history['val_dice_coef']
    jaccard = history.history['val_jaccard_coef']
    hausdorff = history.history['val_hausdorff_distance']
    # val_dice_coef = history.history['val_dice_coef', 'val_jaccard_coef', 'dice_coef']
    # print(val_dice_coef)
    max_val_dice_coef = max(dice)
    max_index = dice.index(max_val_dice_coef)
    jaccard_corr = jaccard[max_index]
    hausdorff_corr = hausdorff[max_index]        
    result = (max_val_dice_coef, jaccard_corr, hausdorff_corr)
    epoch = max_index + 1
    return result, epoch

if __name__== '__main__':
   # if len(sys.argv) < 3:
   #     sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    print(contour_type)
    #training_dataset= sys.argv[2]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 100

    print('Mapping ground truth '+contour_type+' contours to images in train...')

    #train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
    test_ctrs = test_ctrs[0:len(test_ctrs)//10]
    train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))


    #train_ctrs_validation = map_all_contours(TRAIN_CONTOUR_PATH_validation, contour_type, shuffle=True)
    # picked_contours = choose_n_contours (numbers_of_contours, list(train_ctrs_original), "/") 
    print('Done mapping training set')

    # split = int(0*len(a))
    # train_ctrs=a[split:]
    # #dev_ctrs = b[0:split]
    # print(len(a))
    # print("before")

    print(len(train_ctrs))
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)
    # print(np.array(mask_train).shape)
    # print("mask train shape")    (num_ctrs, 100, 100, 1)
    
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(test_ctrs,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    #weights = 'C:\\Users\\user\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=None)

    # sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss= dice_coef_loss,
                  # metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)
    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_train)/mini_batch_size)):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask, sample_weight = 1)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['sunnybrook', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('realtime'):
            os.makedirs('realtime')
        save_path = os.path.join('realtime', save_file)
        #print(save_path)
        model.save_weights(save_path)




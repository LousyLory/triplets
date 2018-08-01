"""Containes a helper class for image input pipelines in tensorflow."""
'modified for triplets'

import tensorflow as tf
import numpy as np
import random
import re

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import cv2

'''
bg_flag = True
if bg_flag:
    from siamese_bg_util import getImageFromWord
else:
    from util import getImageFromWord
'''

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self, mode, batch_size, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.batch_size = batch_size
        self.mode = mode
        #self._read_words()

        if mode == 'training':
            self._read_train_files()
            #self._load_all_files()
        elif mode == 'validation':
            self._read_val_files()
            #self._load_all_files()
        else:
            self._read_test_files()
            #self._load_all_files()

    def _read_image_bgr(self, imagename):
        bgr_img = cv2.imread(imagename)
        return bgr_img

    def get_runtime_batch_from_RAM(self):
        if self.mode == 'training':
            data_len = self.total_train_data
        elif self.mode == 'validation':
            data_len = self.total_val_data
        else:
            data_len = self.total_test_data

	_ids = np.random.permutation(range(data_len))
	_ids = _ids[0:self.batch_size]
        
        left_images = []
	mid_images = []
        right_images = []
        #labels = []
        for i in _ids:
            if self.mode == 'training':
                left_images.append(self._read_image_bgr(self.train_left_words[i]))
		mid_images.append(self._read_image_bgr(self.train_mid_words[i]))
                right_images.append(self._read_image_bgr(self.train_right_words[i]))
                #labels.append(int(self.train_labels[i]))
            elif self.mode == 'validation':
                left_images.append(self._read_image_bgr(self.val_left_words[i]))
		mid_images.append(self._read_image_bgr(self.val_mid_words[i]))
                right_images.append(self._read_image_bgr(self.val_right_words[i]))
                #labels.append(int(self.val_labels[i]))
            else:
                left_images.append(self._read_image_bgr(self.test_left_words[i]))
		mid_images.append(self._read_image_bgr(self.test_mid_words[i]))
                right_images.append(self._read_image_bgr(self.test_right_words[i]))
                #labels.append(int(self.test_labels[i]))


        #self.left_images = np.array(left_images)
	#self.mid_images = np.array(mid_images)
        #self.right_images = np.array(right_images)
        #self.y_labels = np.array(labels)
	return np.array(left_images), np.array(mid_images), np.array(right_images)
        pass
    '''	
    def get_runtime_batch_from_RAM(self):
	self._load_all_files()
	batch_x1 = self.left_images
	batch_x2 = self.mid_images
	batch_x3 = self.right_images
	
        total_data = range(len(self.left_images))
        random.shuffle(total_data)
        ids = total_data[0:self.batch_size]
        batch_x1 = np.copy(self.left_images[ids])
	batch_x2 = np.copy(self.mid_images[ids])
        batch_x3 = np.copy(self.right_images[ids])
        #y = np.copy(self.y_labels[ids])
	
        return np.array(batch_x1), np.array(batch_x2), np.array(batch_x3)
    '''
    def get_all_files(self):
        batch_x1 = np.copy(self.left_images)
	batch_x2 = np.copy(self.mid_images)
        batch_x3 = np.copy(self.right_images)
        #y = np.copy(self.y_labels)

        return np.array(batch_x1), np.array(batch_x2), np.array(batch_x3)        



    def _read_train_files(self):
        '''
        reads train files
        requires you to save the files in a directory called train_txt_files
        '''
	s = '/mnt/nfs/work1/elm/ray/words/images/train/nobg_nopad_txt_files/'
        f = open(s+'image_files.txt', 'rb')
        all_words = f.readlines()
        f.close()
        
	self.total_train_data = len(all_words)
        '''
        strip new line characters
        '''
	all_words = [l.strip('\n\r') for l in all_words]
	self.train_left_words = ['/mnt/nfs/work1/elm/ray/words/images/train/nobg_nopad_left_images/'+x for x in all_words]
	self.train_mid_words = ['/mnt/nfs/work1/elm/ray/words/images/train/nobg_nopad_mid_images/'+x for x in all_words]
	self.train_right_words = ['/mnt/nfs/work1/elm/ray/words/images/train/nobg_nopad_right_images/'+x for x in all_words]


    def _read_val_files(self):
        '''
        reads validation files
        requires you to save the files in a directory called train_txt_files
        '''
	s = '/mnt/nfs/work1/elm/ray/words/images/val/nobg_nopad_txt_files/'
        f = open(s+'image_files.txt', 'rb')
        all_words = f.readlines()
        f.close()

        self.total_val_data = len(all_words)
        '''
        strip new line characters
        '''
	all_words = [l.strip('\n\r') for l in all_words]
        self.val_left_words = ['/mnt/nfs/work1/elm/ray/words/images/val/nobg_nopad_left_images/'+x for x in all_words]
        self.val_mid_words = ['/mnt/nfs/work1/elm/ray/words/images/val/nobg_nopad_mid_images/'+x for x in all_words]
        self.val_right_words = ['/mnt/nfs/work1/elm/ray/words/images/val/nobg_nopad_right_images/'+x for x in all_words]
	
    def _read_test_files(self):
        '''
        reads test files
        requires you to save the files in a directory called train_txt_files
        '''
	s = '/mnt/nfs/work1/elm/ray/words/images/test/nobg_nopad_txt_files/'
        f = open(s+'image_files.txt', 'rb')
        all_words = f.readlines()
        f.close()

        self.total_test_data = len(all_words)
        '''
        strip new line characters
        '''
        all_words = [l.strip('\n\r') for l in all_words]
        self.test_left_words = ['/mnt/nfs/work1/elm/ray/words/images/test/nobg_nopad_left_images/'+x for x in all_words]
        self.test_mid_words = ['/mnt/nfs/work1/elm/ray/words/images/test/nobg_nopad_mid_images/'+x for x in all_words]
        self.test_right_words = ['/mnt/nfs/work1/elm/ray/words/images/test/nobg_nopad_right_images/'+x for x in all_words]

    def get_run_time_batch_from_files(self):
        '''
        gets the files at runtime
        for now this implements vanilla file read
        '''
        if self.mode == 'training':
            total_data = range(np.copy(self.total_train_data))
            random.shuffle(total_data)
            ids = total_data[0:self.batch_size]
            batch_x1 = []
            batch_x2 = []
	    batch_x3 = []
            #y = []
            for i in ids:
                batch_x1.append(self._read_image_bgr(self.train_left_words[i]))
		batch_x2.append(self._read_image_bgr(self.train_mid_words[i]))
                batch_x3.append(self._read_image_bgr(self.train_right_words[i]))
                #y.append(int(self.train_labels[i]))
            pass
        elif self.mode == 'validation':
            total_data = np.copy(self.total_val_data)
            ids = range(total_data)
            batch_x1 = []
            batch_x2 = []
	    batch_x3 = []
            #y = []
            for i in ids:
		batch_x1.append(self._read_image_bgr(self.val_left_words[i]))
                batch_x2.append(self._read_image_bgr(self.val_mid_words[i]))
                batch_x3.append(self._read_image_bgr(self.val_right_words[i]))
                #y.append(int(self.val_labels[i]))
            pass
        else:
            total_data = np.copy(self.total_test_data)
            ids = range(total_data)
            batch_x1 = []
            batch_x2 = []
	    batch_x3 = []
            #y = []
            for i in ids:
		batch_x1.append(self._read_image_bgr(self.test_left_words[i]))
                batch_x2.append(self._read_image_bgr(self.test_mid_words[i]))
                batch_x3.append(self._read_image_bgr(self.test_right_words[i]))
                #y.append(int(self.test_labels[i]))
            pass
	#q = batch_x2
	batch_x1 = np.array(batch_x1)
	batch_x2 = np.array(batch_x2)
	batch_x3 = np.array(batch_x3)
	'''
	if len(batch_x2.shape) == 1:
		print(ids)
		for i in range(batch_x2.shape[0]):
			print(i) 
			print(batch_x2[i].shape)
	'''
	#y = np.array(y)
        return batch_x1, batch_x2, batch_x3

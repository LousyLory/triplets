"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import random
import re

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import cv2

bg_flag = True
if bg_flag:
    from siamese_bg_util import getImageFromWord
else:
    from util import getImageFromWord

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

        self._read_words()

        self._read_train_file()

        # initial shuffling of the file and label lists (together!)
        # if shuffle:
        #     self._shuffle_lists()

        # convert lists to TF tensor
        #self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        #self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        #data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        #self.data = None

    def _read_train_file(self):
        f = open('word_train_with_bg.txt','rb')
        self.train_files_dict = {}
        lines = f.readlines()
        for line in lines:
            image,label = line.split()
            label = int(label)
            if label in self.train_files_dict:
                self.train_files_dict[label].append(image)
            else:
                self.train_files_dict[label] = [image]
        

    def _read_words(self):
        #f = open("words4k.txt")
        #f = open("US_Cities.txt")
    	f = open("balanced_words.txt")
        lines = f.readlines()
        f.close()
        self.words = []
        for line in lines:
            self.words.append(re.sub(r'[^a-zA-Z0-9]','',line))
        self.num_words = len(self.words)

    def _read_image_bgr(self, imagename):
        bgr_img = cv2.imread(imagename)
        return bgr_img

    def get_run_time_batch_from_files(self):
        batch_x1_data = []
        batch_x2_data = []
        batch_y1 = []
        batch_y2 = []

        for i in range(self.batch_size):
            label1 = random.choice(self.train_files_dict.keys())
            label2 = label1
            
            flag = random.randint(0,10)
            if flag>3:
                label2 = random.choice(self.train_files_dict.keys())

                
            batch_x1_data.append(self._read_image_bgr(random.choice(self.train_files_dict[label1])))
            batch_x2_data.append(self._read_image_bgr(random.choice(self.train_files_dict[label2])))
            
            if label1==label2:
                batch_y1.append(0)
                batch_y2.append(0)
            else:
                batch_y1.append(1)
                batch_y2.append(0)

        return np.array(batch_x1_data), np.array(batch_y1), np.array(batch_x2_data), np.array(batch_y2)

    def get_run_time_batch(self):
        batch_x1_data = []
        batch_x2_data = []
        batch_y1 = []
        batch_y2 = []

        for i in range(self.batch_size):
            word1 = self.words[random.randint(0,self.num_words-1)]
            word2 = word1
            x1_label = 0
            x2_label = 0
            flag = random.randint(0,10)
            if flag>3:
                x2_label = 1
                word2 = self.words[random.randint(0,self.num_words-1)]
            batch_x1_data.append(getImageFromWord(word1))
            batch_x2_data.append(getImageFromWord(word2))
            batch_y1.append(x1_label)
            batch_y2.append(x2_label)

        return np.array(batch_x1_data), np.array(batch_y1), np.array(batch_x2_data), np.array(batch_y2)


    def get_test_batch(self, size):
        test_word = self.words[random.randint(0, self.num_words - 1)]
        batch_x1_data = []
        batch_y1 = []
        test_words = []
        batch_x1_data.append(getImageFromWord(test_word))
        test_words.append(test_word)
        batch_y1.append(0)
        for i in range(size-1):
            prob = random.randint(0,10)
            if prob>7:
                batch_x1_data.append(getImageFromWord(test_word))
                batch_y1.append(0)
                test_words.append(test_word)
            else:
                word2 = self.words[random.randint(0,self.num_words-1)]
                batch_x1_data.append(getImageFromWord(word2))
                batch_y1.append(1)
                test_words.append(word2)
        return np.array(batch_x1_data), np.array(batch_y1), test_words



    def get_batch_for_siamese_network(self):
        number_of_labels = len(self.image_dict.keys())
        batch_x1 = []
        batch_x2 = []
        batch_y1 = []
        batch_y2 = []
        for i in range(self.batch_size):
            x1_label = random.randint(0, number_of_labels-1)
            x2_label = x1_label
            flag = random.randint(0,1)
            if flag:
                x2_label = random.choice([label for label in range(number_of_labels) if label != x1_label])
            batch_y1.append(x1_label)
            batch_y2.append(x2_label)
            batch_x1.append(self.image_dict[x1_label][random.randint(0, len(self.image_dict[x1_label]) - 1)])
            batch_x2.append(self.image_dict[x2_label][random.randint(0, len(self.image_dict[x2_label]) - 1)])





        # same_image_count = 0
        # for i in range(self.batch_size):
        #     #print(batch_y1[i], batch_y2[i])
        #     print(batch_x1[i], batch_x2[i])
        #     if batch_y1[i] == batch_y2[i]:
        #         same_image_count+=1
        # print(same_image_count)
        batch_x1_data = []
        batch_x2_data = []
        for i in range(self.batch_size):
            img1 = cv2.imread(batch_x1[i])
            img1 = cv2.resize(img1, (227, 227))
            #print(img1.shape)
            img1 = img1[:,:,::-1]
            batch_x1_data.append(img1)
            img2 = cv2.imread(batch_x2[i])
            img2 = cv2.resize(img2, (227, 227))
            img2 = img2[:, :, ::-1]
            batch_x2_data.append(img2)
        return np.array(batch_x1_data), np.array(batch_y1), np.array(batch_x2_data), np.array(batch_y2)



    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        #one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        #one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label

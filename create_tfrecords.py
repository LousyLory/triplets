import cv2
import numpy as np
import skimage.io as io
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 

mystring = '.png'
f = open('./txt_files/image_files.txt', 'rb')
left_names = f.readlines()
f.close()
left_names = [l.strip('\n\r') for l in left_names]
left_names = [s + mystring for s in left_names]

f = open('./txt_files/word_files.txt', 'rb')
right_names = f.readlines()
f.close()
right_names = [l.strip('\n\r') for l in right_names]
right_names = [s + mystring for s in right_names]

f = open('./txt_files/y_labels.txt', 'rb')
labels = f.readlines()
f.close()
labels = [l.strip('\n\r') for l in labels]

filename_pairs = zip(left_names, right_names, labels)

tfrecords_filename = 'siamese_train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for img_path, annotation_path, _y in filename_pairs:
    
	left_img = cv2.imread(img_path)
	right_img = cv2.imread(annotation_path)
	label = int(_y)
	# we have to know sizes
	# of images to later read raw serialized string,
	# convert to 1d array and convert to respective
	# shape that image used to have.
	height = left_img.shape[0]
	width = left_img.shape[1]
    
	img_raw = left_img.tostring()
	annotation_raw = right_img.tostring()
    
	example = tf.train.Example(features=tf.train.Features(feature={
        	'height': _int64_feature(height),
        	'width': _int64_feature(width),
        	'left_image': _bytes_feature(img_raw),
        	'right_image': _bytes_feature(annotation_raw),
			'labels': _int64_feature(label)}))
    
	writer.write(example.SerializeToString())

writer.close()

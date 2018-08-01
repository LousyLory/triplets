from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import system things
import tensorflow as tf
import numpy as np
import os
import pickle
from util import get_test_results
import cv2
import matplotlib.pyplot as plt

from data_generator_for_siamese import ImageDataGenerator
from datetime import datetime

from helper import get_dataset
import altered_siamese as siamese
import glob
import re

from scipy.spatial import distance

def evaluate(Im, words):
	batch_size = 128
	sess = tf.InteractiveSession()
	siamese_model = siamese.siamese_network(batch_size)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	# load the model
	saver.restore(sess, './trained_models/triplets_with_more_fonts21.ckpt')
	
	dis_vals = []
	for i in range(len(resultant_vectors_left)):
	    dis_vals.append([])
	    for j in range(len(resultant_vectors_right)):
	        dis_vals[i].append(distance.euclidean(resultant_vectors_left[i],resultant_vectors_right[j]))
	
	match = []
	closest_match = []
	for i in range(len(Im)):
	    dis_vals_sub = np.array(dis_vals[i])
	    qualified_ids = dis_vals_sub.argsort()[0:1]	
	    closest_match.append(qualified_ids)
	    if qualified_ids == i:
	    	match.append(1)
	    else:
	    	match.append(0)

	return match, closest_match
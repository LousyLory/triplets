from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import system things
import tensorflow as tf
import numpy as np
import os
from util import get_test_results

from ray_data_generator_for_siamese import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.data import Dataset
from tqdm import tqdm
import altered_siamese as siamese

# prepare data and tf.session
sess = tf.InteractiveSession()
batch_size = 128
# setup siamese network

siamese_model = siamese.siamese_network(batch_size);
train_step = tf.train.AdamOptimizer(0.01).minimize(siamese_model.loss)
saver = tf.train.Saver()

# Place data loading and preprocessing on the cpu
#with tf.device('/cpu:0'):
tr_data = ImageDataGenerator(mode='training',
                                 batch_size=batch_size,
                                 shuffle=True)

tr_data_ev = ImageDataGenerator(mode='training',
                                 batch_size=8448,
                                 shuffle=True)

val_data = ImageDataGenerator(mode='validation',
                                 batch_size=8448,
                                 shuffle=True)


sess.run(tf.global_variables_initializer())
#sess.run(tf.initialize_all_variables())
# start training

siamese_model.load_initial_weights(sess)
num_epochs = 230
new = True

#tr_batch_x1, tr_batch_x2, tr_batch_x3 = tr_data_ev.get_runtime_batch_from_RAM()
#val_batch_x1, val_batch_x2, val_batch_x3 = val_data.get_runtime_batch_from_RAM()
tr_batch_x1, tr_batch_x2, tr_batch_x3 = tr_data_ev.get_run_time_batch_from_files()
val_batch_x1, val_batch_x2, val_batch_x3 = val_data.get_run_time_batch_from_files()

all_loss = []
#tr_acc = []
#val_acc = []
if new:
    for epoch in range(num_epochs):
        for step in range(1000):
            #batch_x1, batch_y1, batch_x2, batch_y2 = tr_data.get_run_time_batch()
            batch_x1, batch_x2, batch_x3 = tr_data.get_run_time_batch_from_files()
            #batch_x1, batch_x2, batch_x3 = tr_data.get_runtime_batch_from_RAM()
	    #val_batch_x1, val_batch_x2, val_y = val_data.get_runtime_batch_from_RAM()
            #print(batch_x1.shape, batch_x2.shape)
            #batch_y = _y.astype('float')
	    #val_y = val_y.astype('float')
            #batch_y = map(float, _y)
            #print(batch_y.shape)
            #print(siamese_model.o1.eval({siamese_model.x1: batch_x1}))
            
            _, loss_v = sess.run([train_step, siamese_model.loss], 
                feed_dict={siamese_model.x1: batch_x1, 
                siamese_model.x2: batch_x2, siamese_model.x3: batch_x3})

	    #op2 = sess.run([siamese_model.op_labels], 
            #    feed_dict={siamese_model.x1: val_batch_x1, 
            #    siamese_model.x2: val_batch_x2, siamese_model.y_: val_y})

	    #v_a = float(np.sum(val_y == op2)) / 128.0
	    #t_a = float(np.sum(batch_y == op1)) / 128.0

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

	    all_loss.append(loss_v)
	    #tr_acc.append(t_a)
	    #val_acc.append(v_a)
            print('iteration %d: loss %.3f' % (step, loss_v))

        # evaluate
        # get all training data
        #tr_batch_x1, tr_batch_x2, tr_y = tr_data.get_runtime_batch_from_RAM()
        #result_vectors = siamese_model.o1.eval({siamese_model.x1: tr_batch_x1})
        #get_test_results(tr_y, result_vectors)
	#tr_batch_x1, tr_batch_x2, tr_y = tr_data_ev.get_runtime_batch_from_RAM()
        #val_batch_x1, val_batch_x2, val_y = val_data.get_runtime_batch_from_RAM()
	v = 0
	t = 0
	count = 0
	for i in range(0,84480,128):
		op1 = sess.run([siamese_model.eval_func],
        	        feed_dict={siamese_model.x1: tr_batch_x1[i:i+128,:,:,:],
                	siamese_model.x2: tr_batch_x2[i:i+128,:,:,:], siamese_model.x3: tr_batch_x3[i:i+128]})
		op2 = sess.run([siamese_model.eval_func],
        	        feed_dict={siamese_model.x1: val_batch_x1[i:i+128,:,:,:],
                	siamese_model.x2: val_batch_x2[i:i+128,:,:,:], siamese_model.x3: tr_batch_x3[i:i+128]})

		v = np.sum(op1) / 128.0
		t = np.sum(op2) / 128.0
		
	v = v/float(count)
	t = t/float(count)
	print('epoch %d: training: %f validation: %f' % (epoch, t, v))


   	if (epoch+1)%11==0:
       	 	saver.save(sess, 'siamese_with_more_fonts'+str(epoch)+'.ckpt')
    #     embed = siamese.o1.eval({siamese.x1: mnist.test.images})
    #     embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')

#np.save('tr_acc.npy', tr_acc)
#np.save('val_acc.npy', val_acc)
np.save('loss.npy', all_loss)
# # visualize result
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)

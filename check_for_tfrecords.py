import tensorflow as tf
import matplotlib.pyplot as plt
from read_tfrecords import read_and_decode

tfrecords_filename = 'siamese_train.tfrecords'

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

# Even when reading in multiple threads, share the filename
# queue.
image, annotation, labels = read_and_decode(filename_queue)
print image.shape, annotation.shape, labels.shape

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Let's read off 3 batches just for example
    for i in xrange(3):
    
        img, anno, _y = sess.run([image, annotation, labels])
        print(img[0, :, :, :].shape)

        print('current batch')
        
        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random

        #io.imshow(img[0, :, :, :])
        plt.imshow(img[0, :, :, :])
        plt.show()

        #io.imshow(anno[0, :, :, 0])
        plt.imshow(anno[0, :, :, :])
        plt.show()

        print _y[0]
        
        #io.imshow(img[1, :, :, :])
        plt.imshow(img[1, :, :, :])
        plt.show()

        #io.imshow(anno[1, :, :, 0])
        plt.imshow(anno[1, :, :, :])
        plt.show()

        print _y[1]
import tensorflow as tf
import numpy as np
import time
input_dim=108
output_dim=7
n_filters = 50
batch_size = 59412
HL1_SIZE=50

w = np.random.random([output_dim, input_dim, n_filters])
feats = np.random.random([batch_size, input_dim])
alpha_gen = np.random.random([batch_size, n_filters])
alpha_dense = alpha_gen / np.reshape(np.sum(alpha_gen, axis= 1), [batch_size, 1])
alpha_gen[alpha_gen >= np.reshape(np.max(alpha_gen, axis= 1),[batch_size, 1] )] = 1
alpha_gen[:, :] = 0




with tf.variable_scope("udi_test_efficiency") as scope:
    x_input = tf.placeholder(tf.float32, shape=(None, input_dim), name='x_input')
    x = tf.stack([x_input for _ in range(output_dim)], axis=0)
    W = tf.placeholder(tf.float32, shape=[output_dim, input_dim, n_filters],
                       name='W')
    alpha = tf.placeholder(tf.float32, shape=(None, n_filters), name='alpha')
    mp = tf.matmul(x, W)  # None x 50
    y_pred = tf.transpose(tf.reduce_sum(tf.multiply(mp, alpha), axis=2))

    print()

with tf.Session() as sess:
    start = time.time()
    yp = sess.run(y_pred, feed_dict= {x_input:feats, W:w, alpha: alpha_gen} )
    stop = time.time()
    print("{0:.4f} s".format(stop-start))
    start = time.time()
    yp = sess.run(y_pred, feed_dict= {x_input:feats, W:w, alpha: alpha_dense} )
    stop = time.time()
    print("{0:.4f} s".format(stop-start))







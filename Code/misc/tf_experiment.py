import tensorflow as tf
import numpy as np
import time
input_dim=108
output_dim=7
n_filters = 50
batch_size = 59412
HL1_SIZE= 60
HL2_SIZE = 50


def regression_with_two_hidden_layers_build_with_batch_normalization(input_dim, output_dim, scope_name, layer1_size = HL1_SIZE,
                                                                     layer2_size = HL2_SIZE):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')
    m1, v1 = tf.nn.moments(x, axes=[0])
    x_normed = tf.nn.batch_normalization(x, mean=m1, variance=v1, offset=None, scale=None, variance_epsilon=0.001)

    with tf.variable_scope(scope_name) as scope: # nn1_h2_reg
        w1 = tf.get_variable("w1", shape=[input_dim, layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('w2', shape = [layer1_size, layer2_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape = [layer2_size], initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable('w3', shape = [layer2_size, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_output = tf.get_variable('b_output', shape=[output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
        h1 = (tf.matmul(x_normed, w1))
        m2, v2 = tf.nn.moments(h1, axes=[0])
        hidden_layer_normed = tf.nn.batch_normalization(h1, mean=m2, variance=v2, offset=b1, scale=None, variance_epsilon=0.001)
        hidden_layer = tf.nn.relu(hidden_layer_normed)
        h2 = tf.matmul(hidden_layer, w2)
        m3, v3 = tf.nn.moments(h2, axes=[0])
        hidden_layer_2_normed = tf.nn.batch_normalization(h2, mean=m3, variance=v3, offset=b2, scale=None, variance_epsilon=0.001)
        hidden_layer_2 = tf.nn.relu(hidden_layer_2_normed)
        y_pred = tf.matmul(hidden_layer_2, w3) + b_output
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, w3, b1, b2)]
        regularizer = tf.add_n(l2_losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred,

regression_with_two_hidden_layers_build_with_batch_normalization(input_dim=160, output_dim=1, scope_name='udi')




#
#
#
# w = np.random.random([output_dim, input_dim, n_filters])
# feats = np.random.random([batch_size, input_dim])
# alpha_gen = np.random.random([batch_size, n_filters])
# alpha_dense = alpha_gen / np.reshape(np.sum(alpha_gen, axis= 1), [batch_size, 1])
# alpha_gen[alpha_gen >= np.reshape(np.max(alpha_gen, axis= 1),[batch_size, 1] )] = 1
# alpha_gen[:, :] = 0
#
#
#
#
# with tf.variable_scope("udi_test_efficiency") as scope:
#     x_input = tf.placeholder(tf.float32, shape=(None, input_dim), name='x_input')
#     x = tf.stack([x_input for _ in range(output_dim)], axis=0)
#     W = tf.placeholder(tf.float32, shape=[output_dim, input_dim, n_filters],
#                        name='W')
#     alpha = tf.placeholder(tf.float32, shape=(None, n_filters), name='alpha')
#     mp = tf.matmul(x, W)  # None x 50
#     y_pred = tf.transpose(tf.reduce_sum(tf.multiply(mp, alpha), axis=2))
#
#     print()
#
# with tf.Session() as sess:
#     start = time.time()
#     yp = sess.run(y_pred, feed_dict= {x_input:feats, W:w, alpha: alpha_gen} )
#     stop = time.time()
#     print("{0:.4f} s".format(stop-start))
#     start = time.time()
#     yp = sess.run(y_pred, feed_dict= {x_input:feats, W:w, alpha: alpha_dense} )
#     stop = time.time()
#     print("{0:.4f} s".format(stop-start))
#
#
#
#
#
#

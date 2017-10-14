"""
This module contains method for constructing several model architectures.
"""
import numpy as np
import tensorflow as tf
from model.model_hyperparams import LEARNING_RATE
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.ml_utils import BestWeightsQueue, Dataset



PRINT_DURING_LEARNING = True

def linear_regression_build(input_dim, output_dim, scope_name):
    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')


    with tf.variable_scope(scope_name) as scope:
        w1 = tf.get_variable("w1", shape=[input_dim, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())

        y_pred = tf.matmul(x, w1) + b1

    return x, y, y_pred


def regression_with_one_hidden_layer_build(input_dim, output_dim, scope_name, layer1_size):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')

    with tf.variable_scope('nn1_hl_reg') as scope:

        w1 = tf.get_variable("w1", shape=[input_dim, layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", shape=[layer1_size, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[output_dim],
                              initializer=tf.contrib.layers.xavier_initializer())
        hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)
        y_pred = tf.matmul(hidden_layer, w2) + b2
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, b1, b2)]
        regularizer = tf.add_n(l2_losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred



def regression_with_two_hidden_layers_build(input_dim, output_dim, scope_name, layer1_size,
                                            layer2_size):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')

    with tf.variable_scope(scope_name) as scope:
        w1 = tf.get_variable("w1", shape=[input_dim, layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('w2', shape = [layer1_size, layer2_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape = [layer2_size], initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable('w3', shape = [layer2_size, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_output = tf.get_variable('b_output', shape=[output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())

        hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)
        hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, w2) + b2)
        y_pred = tf.matmul(hidden_layer_2, w3)  + b_output
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, w3, b1, b2)]
        regularizer = tf.add_n(l2_losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred



def regression_with_two_hidden_layers_build_with_batch_normalization(input_dim, output_dim, scope_name, layer1_size,
                                                                     layer2_size):

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


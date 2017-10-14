"""
This module contains method for constructing and training several  model architectures.
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


def build_loss(y_tensor, y_pred_tensor, scope_name, reg_lambda = 0.0, huber_delta = 1.0):
    """
    creates a loss function based on Huber loss with regularization
    :param y_tensor: actual
    :param y_pred_tensor: predicted
    :param scope_name: scope_name of the variables where the regularization loss should apply
    :param reg_lambda: regularization coefficient
    :param huber_delta: 
    :return: loss function
    """
    loss = tf.losses.huber_loss(labels=y_tensor, predictions=y_pred_tensor, delta=huber_delta)
    regularizer = tf.losses.get_regularization_loss(scope = scope_name)
    loss += regularizer* reg_lambda
    return loss


def train_model(tensors, loss, training, validation, max_epochs, batch_size, scope_name):
    """
    trains a model given training and validation datasets, which are collections of 
    paired inputs (features per vertex) and labels (predicted activation level per vertex).
     
    :param tensors: (x, y, y_pred) which are repectively input placeholder, lanel place holder, and output tensor
    :param loss: a tf.Operation representing the loss function of the model 
    :param training: a DataSet for training
    :param validation: a DataSet fopr validation
    :param max_epochs: (int) 
    :param batch_size: (int) 
    :param scope_name: scope_name
     
    :return: a list of np.arrays representing the learned values for the weights model.
    """

    x, y, y_pred = tensors
    check_every = min(2 *int(np.size(training.data, 0) // batch_size), 200)
    trained_variables = BestWeightsQueue(max_size=8)
    variables = [v for v in tf.trainable_variables() if v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= scope_name)]
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        iter =0
        while training.epochs_completed < max_epochs:
            iter+=1
            x_batch, y_batch = training.next_batch(batch_size= batch_size)
            batch_feed_dict = {x: x_batch, y: y_batch}
            region_feed_dict = {x: training.data, y: training.labels}

            # take a gradient step
            session.run(optimizer, feed_dict=batch_feed_dict)

            if iter % check_every == 0:
                # check root mean sum of square loss on the whole training and validation set
                activation_validation_prediction = session.run(y_pred, feed_dict={x: validation.data})
                training_loss, training_pred = session.run([loss, y_pred], feed_dict=region_feed_dict)
                rmse_training = rmse_loss(training_pred, training.labels)
                rss_validation = rmse_loss(activation_validation_prediction, validation.labels)
                current_weights = [w.eval() for w in variables]
                if PRINT_DURING_LEARNING:
                    print("iteration: {0}, training loss = {1:.2f}, training rmse = {2:.2f}, validation rmse = {3:.2f}".
                          format(iter, training_loss, rmse_training, rss_validation))

                if not trained_variables.update(rss_validation, current_weights):
                    # loss on validation set with current weughts is not among the k-best
                    # this implies convergence on validation.
                    break

    return trained_variables.get_best_weights()
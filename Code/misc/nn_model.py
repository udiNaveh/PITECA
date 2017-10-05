import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.io_utils import *
from sharedutils.subject import *
from model.models import *
import os
import matplotlib.pyplot as plt
import cifti
import sharedutils.general_utils as general_utils
import definitions
from sharedutils.ml_utils import BestWeightsQueue, Dataset

from sharedutils.general_utils import union_dicts
from misc.model_hyperparams import HL1_SIZE, HL2_SIZE, LEARNING_RATE
# hyper parameters



PRINT_DURING_LEARNING = True


def linear_regression_build(input_dim, output_dim, scope_name):
    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')


    with tf.variable_scope(scope_name) as scope:
        w1 = tf.get_variable("w1", shape=[input_dim, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())

        y_pred = tf.matmul(x, w1)

    return x, y, y_pred


def linear_regression_build_old(input_dim, output_dim, scope_name):
    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')
    reg_lambda = tf.placeholder(tf.float32, name='reg_lambda')

    with tf.variable_scope(scope_name) as scope:
        w1 = tf.get_variable("w1", shape=[input_dim, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        y_pred = tf.matmul(x, w1) + b1
        l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        regularizer = tf.add_n(l2_losses)
        loss = tf.reduce_mean(tf.square(y_pred - y)) + reg_lambda * regularizer

    return x, y, y_pred, loss, reg_lambda


def regression_with_one_hidden_layer_build(input_dim, output_dim):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')

    with tf.variable_scope('nn1_hl_reg') as scope:

        w1 = tf.get_variable("w1", shape=[input_dim, HL1_SIZE],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[HL1_SIZE],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", shape=[HL1_SIZE, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[output_dim],
                              initializer=tf.contrib.layers.xavier_initializer())
        hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)
        y_pred = tf.matmul(hidden_layer, w2) + b2
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, b1, b2)]
        regularizer = tf.add_n(l2_losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred


def regression_with_two_hidden_layers_build(input_dim, output_dim, scope_name, layer1_size = HL1_SIZE,
                                            layer2_size = HL2_SIZE):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')

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

        hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)
        hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, w2) + b2)
        y_pred = tf.matmul(hidden_layer_2, w3)  + b_output
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, w3, b1, b2)]
        regularizer = tf.add_n(l2_losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred

def regression_with_two_hidden_layers_build_with_dropout(input_dim, output_dim, scope_name, layer1_size = HL1_SIZE,
                                            layer2_size = HL2_SIZE, dropout_keep_p=1):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')
    false_const = tf.constant(False, dtype=tf.bool)

    is_training = tf.placeholder(dtype=tf.bool)

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

        hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)

        hl1 = tf.layers.dropout(hidden_layer, rate= 1-dropout_keep_p, training=is_training)
        hidden_layer_2 = tf.nn.relu(tf.matmul(hl1, w2) + b2)
        hl2 = tf.layers.dropout(hidden_layer_2, rate= 1-dropout_keep_p, training=is_training)
        y_pred = tf.matmul(hl2, w3)  + b_output
        l2_losses = [tf.nn.l2_loss(v) for v in (w1, w2, w3, b1, b2)]
        regularizer = tf.add_n(l2_losses)

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return x, y, y_pred, is_training


def build_loss(y_tensor, y_pred_tensor, scope_name, reg_lambda = 0.0, huber_delta = 1.0):
    loss = tf.losses.huber_loss(labels=y_tensor, predictions=y_pred_tensor, delta=huber_delta)
    regularizer =  tf.losses.get_regularization_loss(scope = scope_name)
    loss += regularizer* reg_lambda
    return loss


def train_model(tensors, loss, training, validation, max_epochs, batch_size, scope_name):
    x, y, y_pred = tensors
    features, activation = training
    features_validation, activation_validation = validation
    check_every = min(2 *int(np.size(features, 0) // batch_size), 500)
    dataset = Dataset(features, activation)
    n_samples = np.size(features, 0)

    trained_variables = BestWeightsQueue(max_size=6)

    variables = [v for v in tf.trainable_variables() if v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= scope_name)]
    #saver = tf.train.Saver(var_list=tf.trainable_variables())

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        iter =0
        while dataset.epochs_completed < max_epochs:
            iter+=1
            x_batch, y_batch = dataset.next_batch(batch_size= batch_size)
            batch_feed_dict = {x: x_batch, y: y_batch}
            region_feed_dict =  {x: features, y: activation}
            curr_loss, _ = session.run([loss, optimizer], feed_dict=batch_feed_dict)
            if iter % check_every == 0:
                # check residual sum of square loss on validation set
                activation_validation_prediction = session.run(y_pred, feed_dict={x: features_validation})
                training_loss, training_pred = session.run([loss, y_pred], feed_dict=region_feed_dict)
                rss_training = rms_loss(training_pred, activation)
                rss_validation = rms_loss(activation_validation_prediction, activation_validation)
                current_weights = [w.eval() for w in variables]
                if PRINT_DURING_LEARNING:
                    print("iteration: {0}, training loss = {1:.2f}, training rss = {2:.2f}, validation rss = {3:.2f}".format(iter, training_loss, rss_training, rss_validation))

                if not trained_variables.update(rss_validation, current_weights):
                    # current loss is not among the k-best
                    break

    return trained_variables.get_best_weights()

def predict_from_model(tensors, features, saved_weights, scope_name):

    x, y, y_pred = tensors
    variables = [v for v in tf.trainable_variables() if v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= scope_name)]
    assert len(variables)== len(saved_weights)
    weights_feed_dict = {tensor: saved_weights[i] for i, tensor in enumerate(variables)}
    with tf.Session() as session:
        region_feed_dict = union_dicts({x: features}, weights_feed_dict)
        prediction = session.run(y_pred, feed_dict=region_feed_dict)
    return prediction
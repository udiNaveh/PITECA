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

# hyper parameters

hidden_layer_size = 50
hidden_layer_2_size = 50
learning_rate = 0.001

PRINT_DURING_LEARNING = True




def basic_linear_regression_build(n_features, n_tasks):
    x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
    y = tf.placeholder(tf.float32, shape=(None, n_tasks), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(tf.random_normal([n_features,n_tasks]), name = 'w')
        y_pred = tf.matmul(x,w)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    return x, y, y_pred, loss


def regression_with_one_hidden_leyer_build(input_dim, output_dim):

    x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_dim), name='y')
    reg_lambda = tf.placeholder(tf.float32, name = 'reg_lambda')


    with tf.variable_scope('nnreg') as scope:
        # w1 = tf.Variable(tf.random_normal([input_dim, hidden_layer_size]), name ='w1')
        # b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name ='b1')
        # w2 = tf.Variable(tf.random_normal([hidden_layer_size, output_dim]), name ='w2')
        # sc = tf.Variable(tf.random_normal([output_dim]), name ='sc')
        # b2 = tf.Variable(tf.random_normal([output_dim]), name ='b2')

        w1 = tf.get_variable("w1", shape=[input_dim, hidden_layer_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[hidden_layer_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('w2', shape = [hidden_layer_size, hidden_layer_2_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape = [hidden_layer_2_size], initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable('w3', shape = [hidden_layer_2_size, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_output = tf.Variable(tf.random_normal([output_dim]), name ='b_output')

        hidden_layer = tf.nn.tanh(tf.matmul(x, w1) + b1)
        hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, w2) + b2)
        y_pred = tf.matmul(hidden_layer_2, w3)  + b_output
        regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2)
        loss = tf.reduce_mean(tf.square(y_pred-y)) + reg_lambda * regularizer

    return x, y, y_pred, loss, reg_lambda





def learn_one_region(training, validation, max_epochs, batch_size, reg_lambda = 0):

    features, activation = training
    features_validation, activation_validation = validation

    check_every = min(2 *int(np.size(features, 0) // batch_size), 500)
    dataset = Dataset(features, activation)
    n_samples = np.size(features, 0)

    trained_variables = BestWeightsQueue(max_size=6)


    saver = tf.train.Saver()
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    curr_loss = float('inf')
    with tf.Session() as session:
        session.run(init)
        iter =0
        while dataset.epochs_completed < max_epochs:
            iter+=1
            x_batch, y_batch = dataset.next_batch(batch_size= batch_size)
            curr_loss, _ = session.run([loss, optimizer], feed_dict={x: x_batch, y: y_batch})
            if iter % check_every == 0:
                # check residual sum of square loss on validation set
                activation_validation_prediction = session.run(y_pred, feed_dict={x: features_validation})
                training_loss, training_pred = session.run([loss, y_pred], feed_dict={x: features , y: activation})
                rss_training = rms_loss(training_pred, activation)
                rss_validation = rms_loss(activation_validation_prediction, activation_validation)
                current_weights = [w.eval() for w in tf.trainable_variables()]
                if PRINT_DURING_LEARNING:
                    print("iteration: {0}, training loss = {1:.2f}, training rss = {2:.2f}, validation rss = {3:.2f}".format(iter, training_loss, rss_training, rss_validation))

                if not trained_variables.update(rss_validation, current_weights):
                    # current loss is not among the k-best
                    break
                # print("curr_loss = {0:.2f}, avg_loss on last {1} batches = {2:.2f}".format(curr_loss, check_every,next_avg_loss))
        return trained_variables.get_best_weights()


def learn_one_region_2(tensors, training, validation, max_epochs, batch_size, regularization_lambda):

    x, y, y_pred, loss, reg_lambda_placeholder = tensors
    features, activation = training
    features_validation, activation_validation = validation
    check_every = min(2 *int(np.size(features, 0) // batch_size), 500)
    dataset = Dataset(features, activation)
    n_samples = np.size(features, 0)

    trained_variables = BestWeightsQueue(max_size=6)

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    curr_loss = float('inf')
    with tf.Session() as session:
        session.run(init)
        iter =0
        while dataset.epochs_completed < max_epochs:
            iter+=1
            x_batch, y_batch = dataset.next_batch(batch_size= batch_size)
            batch_feed_dict = {x: x_batch, y: y_batch,  reg_lambda_placeholder: regularization_lambda}
            region_feed_dict =  {x: features, y: activation,  reg_lambda_placeholder: regularization_lambda}
            curr_loss, _ = session.run([loss, optimizer], feed_dict=batch_feed_dict)
            if iter % check_every == 0:
                # check residual sum of square loss on validation set
                activation_validation_prediction = session.run(y_pred, feed_dict={x: features_validation})
                training_loss, training_pred = session.run([loss, y_pred], feed_dict=region_feed_dict)
                rss_training = rms_loss(training_pred, activation)
                rss_validation = rms_loss(activation_validation_prediction, activation_validation)
                current_weights = [w.eval() for w in tf.trainable_variables()]
                if PRINT_DURING_LEARNING:
                    print("iteration: {0}, training loss = {1:.2f}, training rss = {2:.2f}, validation rss = {3:.2f}".format(iter, training_loss, rss_training, rss_validation))

                if not trained_variables.update(rss_validation, current_weights):
                    # current loss is not among the k-best
                    break
                # print("curr_loss = {0:.2f}, avg_loss on last {1} batches = {2:.2f}".format(curr_loss, check_every,next_avg_loss))

    return trained_variables.get_best_weights()


def predict_one_region_2(tensors, features, saved_weights):

    x, y, y_pred, loss, reg_lambda_placeholder = tensors
    # saver = tf.train.Saver(var_list=tf.trainable_variables())

    weights_feed_dict = {tensor : saved_weights[i] for i, tensor in enumerate(tf.trainable_variables())}
    prediction = []
    with tf.Session() as session:
        region_feed_dict =  union_dicts({x: features}, weights_feed_dict)
        prediction = session.run(y_pred, feed_dict=region_feed_dict)
    return prediction

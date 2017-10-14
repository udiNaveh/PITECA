"""
This module was used for training models. It is not part of the PITECA gui.
"""
import numpy as np
import tensorflow as tf
from model.data_manager import *
from model.nn_model import *
from model.model_hyperparams import *
from sharedutils.ml_utils import Dataset
import definitions
import sharedutils.general_utils as general_utils
import sharedutils.ml_utils as ml_utils
from sharedutils.io_utils import *

ROI_THRESHOLD = 0.01


# load all data needed
all_features, all_tasks = load_data()
subjects = [Subject(subject_id=general_utils.zero_pad(i + 1, 6)) for i in range(200)]
mem_task_getter = MemTaskGetter(all_tasks, subjects)
spatial_filters_raw, (series, bm) = open_cifti(definitions.ICA_LOW_DIM_PATH)
spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDARD_BM.CORTEX])
soft_filters = softmax(spatial_filters_raw.astype(float) / TEMPERATURE)
soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
soft_filters[:, 2] = 0
soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDARD_BM.N_CORTEX, 1])
hard_filters = np.round(softmax(spatial_filters_raw.astype(float) * 1000))
hard_filters[spatial_filters_raw < SPATIAL_FILTERS_THRESHOLD] = 0
spatial_filters_raw_normalized = demean_and_normalize(spatial_filters_raw, axis=0)


def get_weights_path_by_task(task):
    """
    return a path to save learned weights for task. 
    """
    raise NotImplementedError


def build_loss(y_tensor, y_pred_tensor, scope_name, reg_lambda=0.0, huber_delta=1.0):
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
    regularizer = tf.losses.get_regularization_loss(scope=scope_name)
    loss += regularizer * reg_lambda
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
    check_every = min(2 * int(np.size(training.data, 0) // batch_size), 200)
    trained_variables = BestWeightsQueue(max_size=8)
    variables = [v for v in tf.trainable_variables() if v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)]
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        iter = 0
        while training.epochs_completed < max_epochs:
            iter += 1
            x_batch, y_batch = training.next_batch(batch_size=batch_size)
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


def train_by_roi_and_task(subjects, task, spatial_filters, scope_name):
    """
    trains models (one per each roi) for a given task.
    saves the learned weights
    :param subjects: all subjects in data
    :param task: the task to be learned
    :param spatial_filters: a matrix 
    :param scope_name: the scope name for the variables in computational graph

    """

    global_features = None
    train_subjects, validation_subjects, test_subjects = \
        ml_utils.create_partition(subjects, TRAINING_VALIDATION_TEST_PARTITION)
    learned_weights = {}
    x, y, y_pred = regression_with_two_hidden_layers_build(input_dim=NUM_FEATURES,
                                                           output_dim=1,
                                                           scope_name=scope_name,
                                                           layer1_size=HL2_SIZE,
                                                           layer2_size=HL2_SIZE)

    loss_function = build_loss(y, y_pred, scope_name, reg_lambda=REG_LAMBDA, huber_delta=HUBER_DELTA)
    for j in range(NUM_SPATIAL_FILTERS):
        roi_indices = spatial_filters[: STANDARD_BM.N_CORTEX, j] > ROI_THRESHOLD
        roi_size = np.size(np.nonzero(roi_indices))
        print("train task {} filter {} with {} vertices".format(task.name, j, roi_size))
        if np.size(np.nonzero(roi_indices))<30:
            continue
        roi_features_train, roi_task_train = get_selected_features_and_tasks(
            all_features, train_subjects, roi_indices, task, mem_task_getter, global_features_matrix=global_features)
        roi_features_val, roi_task_val = get_selected_features_and_tasks(
            all_features, validation_subjects, roi_indices, task, mem_task_getter, global_features_matrix=global_features)

        train_data = Dataset(roi_features_train, roi_task_train)
        validation_data = Dataset(roi_features_val, roi_task_val)

        _, weights = train_model(( x, y, y_pred), loss_function, train_data, validation_data,
                                 max_epochs=MAX_EPOCHS_PER_ROI,
                                 batch_size=BATCH_SIZE_R0I,
                                 scope_name=scope_name)

        learned_weights[j] = weights
    save_pickle(learned_weights, get_weights_path_by_task(task))

    return

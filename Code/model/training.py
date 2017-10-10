from model.data_manager import *
from model.nn_model import *
from model.model_hyperparams import *
from sharedutils.ml_utils import Dataset
import definitions
import  sharedutils.general_utils as general_utils

ROI_THRESHOLD = 0.01
all_features, all_tasks = load_data()

all_task_std = np.std(all_tasks, axis=1)
all_task_mean = np.mean(all_tasks, axis=1)
tasks_std = {task: all_task_std[:,i:i+1] for i,task in enumerate(TASKS)}

subjects = [Subject(subject_id=general_utils.zeropad(i+1, 6)) for i in range(200)]
mem_task_getter = MemTaskGetter(all_tasks, subjects)



spatial_filters_raw, (series, bm) = open_cifti(definitions.ICA_LOW_DIM_PATH)
spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDARD_BM.CORTEX])
soft_filters = softmax(spatial_filters_raw.astype(float) * TEMPERATURE)
soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
soft_filters[:, 2] = 0
soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDARD_BM.N_CORTEX, 1])
hard_filters = np.round(softmax(spatial_filters_raw.astype(float) * 1000))
hard_filters[spatial_filters_raw < SPATIAL_FILTERS_THRESHOLD] = 0
spatial_filters_raw_normalized = demean_and_normalize(spatial_filters_raw, axis=0)

NN_WEIGHTS_PATH = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'nn')


def train_by_roi_and_task(subjects, task, spatial_filters, scope_name):
    task_idx = list.index(TASKS, task)
    global_features = np.concatenate((spatial_filters_raw_normalized, demean_and_normalize(all_task_mean[:,task_idx:task_idx+1])), axis=1)
    train_subjects = subjects[:100]
    validation_subjects = subjects[100:130]
    learned_weights = {}
    x, y, y_pred = regression_with_two_hidden_layers_build(input_dim=NUM_FEATURES+NUM_SPATIAL_FILTERS+1, output_dim=1, scope_name='nnhl2fsf')
    loss_function = build_loss(y, y_pred, 'nnhl2fsf', reg_lambda=REG_LAMBDA, huber_delta=HUBER_DELTA)
    try:
        for j in range(NUM_SPATIAL_FILTERS): #
            roi_indices = spatial_filters[: STANDARD_BM.N_CORTEX, j] > ROI_THRESHOLD
            print("train task {} filter {} with {} vertices".format(task.name, j, np.size(np.nonzero(roi_indices))))
            #print("train all tasks filter {} with {} vertices".format( j, np.size(np.nonzero(roi_indices))))
            if np.size(np.nonzero(roi_indices))<30:
                continue
            roi_features_train, roi_task_train = get_selected_features_and_tasks(
                all_features, train_subjects, roi_indices, task, mem_task_getter, global_features_matrix=global_features)
            roi_features_val, roi_task_val = get_selected_features_and_tasks(
                all_features, validation_subjects, roi_indices, task, mem_task_getter, global_features_matrix=global_features)

            np.concatenate([tasks_std[task][roi_indices] for i in range(len(train_subjects))])
            train_data = Dataset(roi_features_train, roi_task_train, np.concatenate([tasks_std[task][roi_indices] for i in range(len(train_subjects))]))
            validation_data = Dataset(roi_features_val, roi_task_val)

            strt = time.time()
            _, weights = train_model(( x, y, y_pred), loss_function, train_data, validation_data, max_epochs=MAX_EPOCHS_PER_ROI,
                                     batch_size=BATCH_SIZE_R0I,
                                     scope_name=scope_name)

            learned_weights[j] = weights

    except Exception as ex:
        print(ex)
    finally:
        pickle.dump(learned_weights, general_utils.safe_open(
            os.path.join(NN_WEIGHTS_PATH, "nn_2hl_by_roi_fsf_check3_100s_weights_{0}_all_filters.pkl".
                         format(task.full_name)), 'wb'))

    #pickle.dump(learned_weights, safe_open(os.path.join(NN_WEIGHTS_PATH, 'all_tasks',"nn_2hl_by_roi_check_70s_weights_all_tasks_all_filters.pkl"), 'wb'))


    return

for task in TASKS:
    train_by_roi_and_task(subjects, task, hard_filters, scope_name='nnhl2fsf')
    tf.reset_default_graph()

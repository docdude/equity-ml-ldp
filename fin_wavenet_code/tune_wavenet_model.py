# A basic tutorial in how I load data and train a model
import tune_wavenet
import utils
import utils_plot
import utils_train
import learning_data
import os
import random as rn
import tensorflow as tf
import numpy as np
import pickle
import datetime
from tensorboard.plugins.hparams import api as hp
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from utils_train import CombinedMetricCallback, CombinedEarlyStopping
from utils_peak_detect_v2 import EnhancedPeakConsistencyCallback
#tf.config.run_functions_eagerly(True)  # Run eagerly for CPU
#tf.data.experimental.enable_debug_mode()  # Help identify input pipeline issues
# A path to re-sampled recordings which are organized into folders by user name.
data_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/data_modified_users_unfiltered'

# Path to where we want to save the training results
run_name = 'tune_wavenet_2_unfiltered_zeronorm_axisnone_combine_bfce'
base_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2'
save_path = os.path.join(base_path, f'run_{run_name}')

# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
#users = ['2','6','7','11']
users_test = ['40']

# List of users we want to train a model for
#users_test = ['2','6','7','11']

# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],
                   'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
                   'time_scale_factors':   [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                                                        # recording with these factors.
                   'stroke_range':         None,       # Augments stroke labels in the dataset to include a range around detected peaks
                   'stroke_sigma':          2.0,
                   'win_len':              180,     # The length of the segmentation window in number of samples
                   'slide_len':            30,      # The slide length used for segmentation
                   'window_normalization': 'tanh_scaled',   # How we want to normalize the windows. Statistical means
                                                            # zero-mean and unit variance for each signal
                   'label_type':           'majority',  # How we label windows.
                   'majority_thresh':      0.75,    # Proportion of samples in a window that have to have the same label
                   'validation_set': {
                        0: 1,  # Transition (NULL + TURN combined as 0)
                        1: 1,  # Freestyle
                        2: 1,  # Breaststroke
                        3: 1,  # Backstroke
                        4: 1,  # Butterfly
                   },
                   'debug':               False,
                   'debug_plot':           False
                   }

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path. Recordings are stored under
# swimming_data.data_dict['original][user_name][recording_name] which is a Pandas DataFrame
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])
# Combine labels moved after compile_windows to allow style 0 normaliztion(axis=none)
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Augments stroke labels in the dataset to include a range around detected peaks
#swimming_data.augment_stroke_labels(stroke_range=data_parameters['stroke_range'])
swimming_data.augment_stroke_labels_to_heatmaps(stroke_sigma=data_parameters['stroke_sigma'])

# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'one_hot_label' or 'sparse_label' or 'stroke_label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])


# Parameters for training the CNN model
training_parameters = {'swim_style_lr':   0.00002,  # Constant for swim style
                       'stroke_lr':       0.00002,
                    #   'common_lr':       2.0e-5,
                       'beta_1':          0.9,
                       'beta_2':          0.999,
                       'batch_size':      128,
                       'max_epochs':      150,      # Keeping small for quick testing
                       'steps_per_epoch': 100,      # Keeping small for quick testing
                       'noise_std':       0.01,    # Noise standard deviation for data augmentation
                       'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                       'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                   # window in the mini-batch
                       'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                       'sample_weights':   'sklearn',    # Sample weights for validation set, 'sklearn' or 'inverse_freq'
                       'stroke_mask':     False,    # Whether to use a mask for stroke labels
                       'labels':          swimming_data.labels,
                       'stroke_labels':   swimming_data.stroke_labels,
                       'stroke_label_output':       True,
                       'swim_style_output':         True,
                       'output_bias':               None,
                       'compute_sample_weights':    True
                       }

# The input shape of the Wavenet model
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']))


# Train all models
for (i, user_test) in enumerate(users_test):
    # Random seed stuff. Maybe this is overkill
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(1337)
    np.random.seed(1337)
    tf.random.set_seed(1337)

    # Path for saving results
    print("Running experiment: %s" % user_test)
    experiment_save_path = os.path.join(save_path, user_test)

    # A path to log directory for Tensorboard
    log_dir = f"logs/tune/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run_name}"
    if os.path.exists(experiment_save_path):
        print(f"Skipping: {user_test}, already processed.")
        continue
    else:
        os.makedirs(experiment_save_path)  # Creates all parent directories if they don't exist

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Manually-defined val_dict for hyperparameter tuning
    val_dict = {
        0: ['28'],  # Transition (challenging)
        1: ['15', '18', '12'],  # Crawl (easy and challenging)
        2: ['28', '39', '32'],  # Breaststroke (easy and challenging)
        3: ['21','22', '35'],  # Backstroke (easy)
        4: ['17', '32']  # Butterfly (challenging)
    }
    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'],
                                                              manual_val_dict=val_dict)
  # if data_parameters['debug']:
    print("Training dictionary: %s" % train_dict)   
    print("Validation dictionary: %s" % val_dict)


    # Calculate stroke label distribution for training set (excluding label 0)
    training_probabilities, training_mean, training_bias, training_stroke_sample_weights, training_stroke_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=None
    )
    training_parameters['output_bias'] = training_bias
    # Calculate stroke label distribution for validation set (excluding label 0)
    validation_probabilities, validation_mean, validation_bias, validation_stroke_sample_weights, validation_stroke_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=val_dict,
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=None
    )

    # Calculate swim style statistics for training and validation sets
    train_counts, train_stats, training_style_sample_weights, training_style_class_weights = utils.calculate_swim_style_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=None)

    val_counts, val_stats, validation_style_sample_weights, validation_style_class_weights = utils.calculate_swim_style_distribution(
        label_user_dict=val_dict, 
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=None)

    # Compare distributions between training and validation sets
    if train_stats and val_stats:
        print("\nDistribution Comparison (Training vs Validation):")
        for label in set(train_stats["label_counts"].keys()) | set(val_stats["label_counts"].keys()):
            train_pct = train_stats["distribution"].get(label, 0) * 100
            val_pct = val_stats["distribution"].get(label, 0) * 100
            diff = abs(train_pct - val_pct)
            print(f"  Swim Style {label}: Train {train_pct:.2f}% vs Val {val_pct:.2f}% (Diff: {diff:.2f}%)")


    #folded_weights_for_user = swimming_data.compute_folded_weights_per_user(train_dict=train_dict)
    #folded_weights_for_user = utils.smooth_folded_weights_dict(folded_weights_for_user, method='log1p', normalize=False)
    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=training_parameters['batch_size'],
                                              noise_std=training_parameters['noise_std'],
                                              mirror_prob=training_parameters['mirror_prob'],
                                              random_rot_deg=training_parameters['random_rot_deg'],
                                              use_4D=False,
                                              swim_style_output=training_parameters['swim_style_output'], 
                                              stroke_label_output=training_parameters['stroke_label_output'],
                                              return_stroke_mask=training_parameters['stroke_mask'],
                                              compute_sample_weights=training_parameters['compute_sample_weights'])
                                              #global_swim_weights=training_style_class_weights,
                                              #global_stroke_weights=training_stroke_class_weights,
                                             # folded_weights_for_user=folded_weights_for_user)

    # Get the validation data with weights and mask
    x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask = swimming_data.get_windows_dict(
        val_dict, return_weights=True, return_mask=True, transition_label=0
    )
 
    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        validation_data = (x_val, {'swim_style_output': y_val_cat, 'stroke_label_output': y_stroke_val},
                                {'swim_style_output': validation_style_sample_weights, 'stroke_label_output': (val_stroke_mask 
                                    if training_parameters['stroke_mask'] 
                                    else validation_stroke_sample_weights)})
        # Set up the combined early stopping callback
        #callbacks = utils_train.get_callbacks_regression(experiment_save_path, user_test, log_dir, x_val, y_val_cat, y_stroke_val)
        #callbacks=[utils_train.CombinedAccuracyF1Metrics(x_val, y_val_cat, y_stroke_val, val_sample_weights=validation_stroke_sample_weights)]
        # Set up the combined early stopping callback
        callbacks = [
            CombinedMetricCallback(alpha=0.7),
            CombinedEarlyStopping(
                monitor1='val_stroke_label_output_weighted_f1_score',
                monitor2='val_swim_style_output_weighted_categorical_accuracy',
                mode1='max',
                mode2='max',
                patience=20,
                restore_best_weights=True
            ),
            EnhancedPeakConsistencyCallback(
                x_val,
                y_stroke_val,
                target_threshold=0.8
            )#,
           # utils_train.CombinedAccuracyF1Metrics(x_val, y_val_cat, y_stroke_val, val_sample_weights=validation_stroke_sample_weights)
        ]

    elif training_parameters['swim_style_output']:
        validation_data = (x_val, y_val_cat, val_sample_weights)
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_categorical_accuracy', patience=10, restore_best_weights=True, mode='max')]
        callbacks.append(utils_train.get_callbacks(experiment_save_path, user_test, log_dir))
    else:
        validation_data = (x_val,  y_stroke_val, (val_stroke_mask if training_parameters['stroke_mask'] else validation_stroke_sample_weights))
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=10, restore_best_weights=True, mode='max')]
        callbacks.append(utils_train.get_callbacks(experiment_save_path, user_test, log_dir))
    
    best_model, best_hps = tune_wavenet.run_hyperparameter_tuning(
            input_shape, 
            nfilt=32,
            data_parameters=data_parameters, 
            training_parameters=training_parameters,
            gen=gen,  
            validation_data=validation_data,
            run_name=run_name,
            callbacks=callbacks
    )

    # Save model in h5 format
    model_h5_path = os.path.join(experiment_save_path, f'model_{user_test}.h5')
    best_model.save(model_h5_path)
    # Save model in native Keras format
    model_keras_path = os.path.join(experiment_save_path, f'model_{user_test}.keras')
    best_model.save(model_keras_path)
    # Saving the history and parameters
    with open(os.path.join(experiment_save_path, 'best_hyperparameters.pkl'), 'wb') as f:
        pickle.dump(best_hps.values, f)  # Save the hyperparameters as a dictionary
    with open(os.path.join(experiment_save_path, 'train_val_dicts.pkl'), 'wb') as f:
        pickle.dump([train_dict, val_dict], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)

    with open(os.path.join(experiment_save_path, 'training_parameters.pkl'), 'wb') as f:
        pickle.dump([training_parameters], f)
  
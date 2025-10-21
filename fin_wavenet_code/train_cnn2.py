# A basic tutorial in how I load data and train a model
import cnn_vanilla1 #LSTM
import cnn_vanilla2 #TimeDistibuted layer wrapping dense
import cnn_vanilla3 #2 dense layers after main backbone
import cnn_vanilla4 #single dense layer after main backbone
import cnn_vanilla5 #convtranspose layer to upsample no dense conv1d sigmoid final
import cnn_vanilla6 #single dense layer using swim_style_branch final connected dense
import cnn_vanilla7 # parallel branches
import cnn_vanilla8 # parallel branches with weighted loss
import cnn_vanilla_dual # parallel branches with weighted loss
import test_lstm
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

#tf.config.run_functions_eagerly(True)  # Run eagerly for CPU
#tf.data.experimental.enable_debug_mode()  # Help identify input pipeline issues
# A path to re-sampled recordings which are organized into folders by user name.
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'

# Path to where we want to save the training results
run_name = 'stroke_25_weighted'
base_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2'
save_path = os.path.join(base_path, f'run_{run_name}')

# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
users = ['2','6','7','11']
users_test = users

# List of users we want to train a model for
#users_test = ['2','6','7','11']

# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],#, 'MAG_0',
                                      #      'MAG_1', 'MAG_2', 'PRESS', 'LIGHT'],    # The sensor data we want to load
                                       #     'MAG_1', 'MAG_2'],    # The sensor data we want to load
                   'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
                   'time_scale_factors':   [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                                                        # recording with these factors.
                   'stroke_range':         6,       # Augments stroke labels in the dataset to include a range around detected peaks
                   'win_len':              180,     # The length of the segmentation window in number of samples
                   'slide_len':            30,      # The slide length used for segmentation
                   'window_normalization': 'statistical',   # How we want to normalize the windows. Statistical means
                                                            # zero-mean and unit variance for each signal
                   'label_type':           'majority',  # How we label windows.
                   'majority_thresh':      0.75,    # Proportion of samples in a window that have to have the same label
                   'validation_set': {
                        0: 1,  # Transition (NULL + TURN combined as 0)
                        1: 1,  # Freestyle
                        2: 1,  # Backstroke
                        3: 1,  # Breaststroke
                        4: 1,  # Butterfly
                        'stroke_labels': ['stroke_labels'],  # Track stroke labels separately
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
# Combine labels
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Augments stroke labels in the dataset to include a range around detected peaks
swimming_data.augment_stroke_labels(stroke_range=data_parameters['stroke_range'])

# Debug visualization time auementation of sensor data and stroke data
if data_parameters['debug_plot']:
    utils_plot.plot_style_specific_axes(swimming_data)


# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'one_hot_label' or 'sparse_label' or 'stroke_label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# Parameters for the CNN model

"""
Get parameters for swim style branch
"""
swim_model_parameters = {
    'filters':        [64, 64, 64, 64],
    'kernel_sizes':   [3, 3, 3, 3],
    'strides':        [None, None, None, None],
    'max_pooling':    [3, 3, 3, 3],
    'units':          [128],
    'activation':     ['elu', 'elu', 'elu', 'elu', 'elu'],
    'batch_norm':     [False, False, False, False, False],
    'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
    'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
    'l2_reg':         [None, None, None, None, None],
    'labels':         [0, 1, 2, 3, 4]
}

"""
Get parameters for stroke detection branch
"""
stroke_model_parameters = {
    'filters':        [64, 64],  # First conv, attention conv
    'kernel_sizes':   [1, 3],    # 1x1 for attention, 3x1 for conv
    'strides':        [None, None],
    'max_pooling':    [None, None],
    'units':          [32],      # LSTM units
    'activation':     ['elu', 'sigmoid', 'sigmoid'],  # conv, attention, output
    'batch_norm':     [False, False, False],
    'drop_out':       [0.3, 0.3, 0.5],  # Adjusted dropout rates
    'max_norm':       [0.5, 0.5, 4.0],  # Reduced max norm constraints

    'l2_reg':         [1e-4, 1e-4, 1e-4],  # Add light L2 regularization
    'labels':         ['stroke_labels']     # binary stroke detection
}

#swim_model_parameters = cnn_vanilla_dual.get_default_swim_model_parameters()
#stroke_model_parameters = cnn_vanilla_dual.get_default_stroke_model_parameters()

# Parameters for training the CNN model
training_parameters = {'swim_style_lr': 0.0005,  # Constant for swim style
                       'stroke_lr': {
                            'initial_lr': 0.0001,
                            'decay_steps': 1000,
                            'decay_rate': 0.90
                        },
                       'beta_1':          0.9,
                       'beta_2':          0.999,
                       'batch_size':      64,
                       'max_epochs':      100,      # Keeping small for quick testing
                       'steps_per_epoch': 100,      # Keeping small for quick testing
                       'noise_std':       0.01,    # Noise standard deviation for data augmentation
                       'mirror_prob':     None,     # Probability of reversing a window for data augmentation
                       'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                   # window in the mini-batch
                       'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                       'sample_weights':   'sklearn',    # Sample weights for validation set, 'sklearn' or 'inverse_freq'
                       'stroke_mask':     False,    # Whether to use a mask for stroke labels
                       'labels':          swimming_data.labels,
                       'stroke_labels':   swimming_data.stroke_labels,
                       'stroke_label_output':     True,
                       'swim_style_output':      False
                       }

# The input shape of the CNN
#input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + len(data_parameters['stroke_labels']), 1)
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']), 1)


# Train all models
for (i, user_test) in enumerate(users_test):
    # Random seed stuff. Maybe this is overkill
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    tf.random.set_seed(1337)

    # Path for saving results
    print("Running experiment: %s" % user_test)
    experiment_save_path = os.path.join(save_path, user_test)

    # A path to log directory for Tensorboard
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{user_test}-{run_name}"
    if os.path.exists(experiment_save_path):
        print(f"Skipping: {user_test}, already processed.")
        continue
    else:
        os.makedirs(experiment_save_path)  # Creates all parent directories if they don't exist

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'])
    if data_parameters['debug']:
        print("Training dictionary: %s" % train_dict)   
        print("Validation dictionary: %s" % val_dict)


    # Calculate stroke label distribution for training set (excluding label 0)
    training_probabilities, training_mean, training_bias, training_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=None
    )

    # Calculate stroke label distribution for validation set (excluding label 0)
    validation_probabilities, validation_mean, validation_bias, validation_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=val_dict,
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=None
    )

    # Optional: Visualize the distributions
    if data_parameters['debug_plot']:
        utils_plot.plot_label_distribution(training_probabilities, validation_probabilities)    
        feature_indices = [0, 1, 2, 3, 4, 5]  # Example: indices of x, y, z axes for accelerometer

        utils_plot.analyze_feature_distribution(train_dict, swimming_data, feature_indices, data_type="training", exclude_label=0)
        utils_plot.analyze_feature_distribution(val_dict, swimming_data, feature_indices, data_type="validation", exclude_label=0)

    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=training_parameters['batch_size'],
                                              noise_std=training_parameters['noise_std'],
                                              mirror_prob=training_parameters['mirror_prob'],
                                              random_rot_deg=training_parameters['random_rot_deg'],
                                              use_4D=True,
                                              swim_style_output=training_parameters['swim_style_output'], 
                                              stroke_label_output=training_parameters['stroke_label_output'],
                                              return_stroke_mask=training_parameters['stroke_mask'])

    if data_parameters['debug']:
        if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
            # Retrieve and print a batch for inspection
            batch_data, batch_outputs, batch_sample_weights = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Swim Style Output Shape: {batch_outputs['swim_style_output'].shape}")
            print(f"Stroke Label Output Shape: {batch_outputs['stroke_label_output'].shape}")
            print(f"Stroke Label Output Unique Values: {np.unique(batch_outputs['stroke_label_output'])}")

            # Debugging stroke mask
            print(f"Stroke Mask Shape: {batch_sample_weights['stroke_label_output'].shape}")
            print(f"Stroke Mask Unique Values: {np.unique(batch_sample_weights['stroke_label_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features + Stroke Labels): {batch_data[example_index, :, :]}")
            print(f"Stroke Labels (from Combined): {batch_outputs['stroke_label_output'][example_index, :, :]}")
            print(f"Stroke Mask (for Combined): {batch_sample_weights['stroke_label_output'][example_index, :, :]}")
        elif training_parameters['swim_style_output']:
            batch_data, batch_outputs = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Swim Style Output Shape: {batch_outputs['swim_style_output'].shape}")
            print(f"Swim Style Output Unique Values: {np.unique(batch_outputs['swim_style_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features): {batch_data[example_index, :, :]}")
            print(f"Swim Style Labels: {batch_outputs['swim_style_output'][example_index, :]}")
        else:
            # Retrieve and print a batch for inspection
            batch_data, batch_outputs, batch_sample_weights = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Stroke Label Output Shape: {batch_outputs['stroke_label_output'].shape}")
            print(f"Stroke Label Output Unique Values: {np.unique(batch_outputs['stroke_label_output'])}")

            # Debugging stroke mask
            print(f"Stroke Mask Shape: {batch_sample_weights['stroke_label_output'].shape}")
            print(f"Stroke Mask Unique Values: {np.unique(batch_sample_weights['stroke_label_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features): {batch_data[example_index, :, :]}")
            print(f"Stroke Labels: {batch_outputs['stroke_label_output'][example_index, :, :]}")
            print(f"Stroke Mask: {batch_sample_weights['stroke_label_output'][example_index, :, :]}")
            # Usage:
            # Plot first sample in batch
            utils_plot.plot_batch_and_mask(batch_data, batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], sample_idx=0)

            # Plot a sample with strokes (if you know the index)
            # Find a sample with strokes
            stroke_samples = np.where(np.any(batch_outputs['stroke_label_output'] == 1, axis=1))[0]
            if len(stroke_samples) > 0:
                print(f"\nPlotting sample with strokes (index {stroke_samples[0]}):")
                utils_plot.plot_batch_and_mask(batch_data, batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], sample_idx=stroke_samples[0])


    # Optimizer
   #optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['lr'], beta_1=training_parameters['beta_1'],
    #                                  beta_2=training_parameters['beta_2'])

    # Get the validation data with weights and mask
    x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask = swimming_data.get_windows_dict(
        val_dict, return_weights=True, return_mask=True, transition_label=0
    )
    # Flatten y_stroke_val to a 1D array
    y_stroke_flat = y_stroke_val.flatten()  
    weights = compute_class_weight('balanced', classes=np.unique(y_stroke_flat), y=y_stroke_flat)
    class_weights = dict(enumerate(weights))
    print(f"All label Class weights: {class_weights}")

    val_stroke_weights = utils.create_balanced_sampler(y_stroke_val, method='sklearn')
    print(f"Sample weights: {val_stroke_weights}")
    # Adjust val_sample_weights shape to match val_stroke_weights for broadcasting
    combined_weights = val_sample_weights[:, np.newaxis, np.newaxis] * val_stroke_weights
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))

    callbacks = utils_train.get_callbacks(experiment_save_path, user_test, log_dir)
    # The cnn_vanilla module contains contains everything to generate the CNN model
    model = cnn_vanilla_dual.cnn_model(input_shape, swim_model_parameters, stroke_model_parameters, training_parameters, use_seed=True, output_bias=training_bias)

    model.summary()

    model = utils_train.compile_model(data_parameters, model, training_parameters, class_weights=weights)



    # Print validation data shapes
    if data_parameters['debug']:
        print(f"x_val shape: {x_val.shape}")
        print(f"y_val_cat shape: {y_val_cat.shape}")
        print(f"y_stroke_val shape: {y_stroke_val.shape}")
        print(f"y_val_sparse shape: {y_val_sparse.shape}")
        print(f"val_sample_weights shape: {val_sample_weights.shape}")
        print(f"val_stroke_mask shape: {val_stroke_mask.shape}")
        # Check unique values in stroke mask
        print(f"y_val_cat unique values: {np.unique(y_val_cat)}")
        print(f"y_stroke_val unique values: {np.unique(y_stroke_val)}")
        print(f"y_val_sparse unique values: {np.unique(y_val_sparse)}")
        print(f"val_stroke_mask unique values: {np.unique(val_stroke_mask)}")
        # Compare combined stroke data with stroke labels and mask
        example_index = 0  # Choose a batch sample index to debug
        print("\nSample Debug:")
        print(f"x_val (Sensor Features): {x_val[example_index, :, :]}")
        print(f"y_stroke_val (Stroke Labels): {y_stroke_val[example_index, :, :]}")
        print(f"y_val_cat: {y_val_cat[example_index,  :]}")
        print(f"y_val_sparse: {y_val_sparse[example_index]}")
        print(f"val_stroke_mask: {val_stroke_mask[example_index, :, :]}")
   # batch_data, batch_outputs, batch_sample_weights = next(gen)

    #utils.verify_masking(batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], "Batch", batch_size=64)
    #utils.verify_masking(y_stroke_val, val_stroke_mask, "Validation", batch_size=64)

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        validation_data = (x_val, {'swim_style_output': y_val_cat, 'stroke_label_output': y_stroke_val},
                                {'swim_style_output': val_sample_weights, 'stroke_label_output': (val_stroke_mask 
                                    if training_parameters['stroke_mask'] 
                                    else val_stroke_weights)})
    elif training_parameters['swim_style_output']:
        validation_data = (x_val, {'swim_style_output': y_val_cat},
                        {'swim_style_output': val_sample_weights})
    else:
        validation_data = (x_val, {'stroke_label_output': y_stroke_val},
                            {'stroke_label_output': (val_stroke_mask 
                                                if training_parameters['stroke_mask'] 
                                                else combined_weights)})
    
        # Pass to model.fit
    history = model.fit(
        gen,  
        validation_data=validation_data,
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
        #callbacks=[checkpoint_swim_style, checkpoint_stroke_label, tensorboard_callback]
        callbacks=callbacks,
        verbose=1
    )

    # Save model in h5 format
    model_h5_path = os.path.join(experiment_save_path, f'model_{user_test}.h5')
    model.save(model_h5_path)
    # Save model in native Keras format
    model_keras_path = os.path.join(experiment_save_path, f'model_{user_test}.keras')
    model.save(model_keras_path)
    # Saving the history and parameters
    with open(os.path.join(experiment_save_path, 'train_val_dicts.pkl'), 'wb') as f:
        pickle.dump([train_dict, val_dict], f)
    with open(os.path.join(experiment_save_path, 'history.pkl'), 'wb') as f:
        pickle.dump([history.history], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)
    with open(os.path.join(experiment_save_path, 'swim_model_parameters.pkl'), 'wb') as f:
        pickle.dump([swim_model_parameters], f)
    with open(os.path.join(experiment_save_path, 'stroke_model_parameters.pkl'), 'wb') as f:
        pickle.dump([stroke_model_parameters], f)
    with open(os.path.join(experiment_save_path, 'training_parameters.pkl'), 'wb') as f:
        pickle.dump([training_parameters], f)


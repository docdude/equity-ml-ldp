import tensorflow as tf
from keras_tuner import HyperModel, BayesianOptimization
from utils_train import compile_model,  CombinedEarlyStopping
import keras_tuner as kt
import pickle
import os
import datetime
from wavenet_model import build_wavenet_model

def get_default_training_parameters():
    """
    Get a default set of parameters used to train a cnn model
    :return: A dictionary containing parameter names and values
    """
    training_parameters = {'swim_style_lr': 0.0005,  # Constant for swim style
                        'stroke_lr': 0.0005,
                        'beta_1':          0.9,
                        'beta_2':          0.999,
                        'batch_size':      64,
                        'max_epochs':      48,      # Keeping small for quick testing
                        'steps_per_epoch': 100,      # Keeping small for quick testing
                        'noise_std':       0.01,    # Noise standard deviation for data augmentation
                        'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                        'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                    # window in the mini-batch
                        'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                        'labels':          [0, 1, 2, 3, 4],
                        'stroke_labels': ['stroke_labels'],  # Labels for stroke predictions
                        'stroke_label_output':      True,
                        'swim_style_output':        True,
                        'output_bias':              None
                        }
    return training_parameters

def create_loss_parameters(hp):
    return {
        'alpha': [hp.Float(f'alpha', min_value=0.75, max_value=0.95, step=0.05)],
        'gamma': [hp.Float(f'gamma', min_value=1.0, max_value=2.0, step=0.05)],  
        'target_height': [hp.Float(f'target_height', min_value=0.80, max_value=0.90, step=0.05)],
        'peak_weight': [hp.Float(f'peak_weight', min_value=4.0, max_value=15.0, step=1.0)],
        'peak_threshold': [hp.Float(f'peak_threshold', min_value=0.55, max_value=0.70, step=0.05)]
    }


class InvalidHyperparameterConfiguration(Exception):
    """Custom exception to indicate invalid hyperparameter configuration."""
    pass

class SwimStrokeHyperModel(HyperModel):
    def __init__(self, input_shape, nfilt, training_parameters, data_parameters):
        self.input_shape = input_shape
        self.nfilt = nfilt
        self.training_parameters = training_parameters
        self.data_parameters = data_parameters

    def build(self, hp):
        # Create hyperparameters
        loss_parameters = create_loss_parameters(hp)

         # Add learning rate tuning
        swim_style_lr = hp.Float('swim_style_lr', min_value=1e-6, max_value=1e-3, sampling='log')

        stroke_lr = hp.Float('stroke_lr', min_value=1e-6, max_value=1e-3, sampling='log')


        self.training_parameters['swim_style_lr'] = swim_style_lr
        self.training_parameters['stroke_lr']= stroke_lr

        try:
            # Validate hyperparameters before model creation
           # validate_model_parameters(swim_model_parameters, stroke_model_parameters, self.input_shape)

            # Build the CNN model
            model = build_wavenet_model(
                input_shape=self.input_shape,
                num_styles=len(self.training_parameters['labels']),
                nfilt=self.nfilt, 
                output_bias=self.training_parameters['output_bias']
            )

            # Compile the model
            model = compile_model(
                data_parameters=self.data_parameters,
                model=model,
                training_parameters=self.training_parameters,
                loss_parameters=loss_parameters
            )

            return model
        
        except (ValueError, InvalidHyperparameterConfiguration) as e:
            print(f"Error in model building: {e}. Skipping this trial.")
            raise Exception(f"TunerError: {str(e)}")  # Re-raise as a generic Exception with a specific message


class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Logs:")
        for key, value in logs.items():
            print(f"{key}: {value}")

def get_tensorboard_callback(log_dir):
    """
    Create a TensorBoard callback for logging hyperparameters and metrics.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log histograms of weights every epoch
        write_graph=True,  # Log the computation graph
        write_images=True,  # Log model weights as images
        update_freq='epoch',  # Log metrics at the end of each epoch
        profile_batch=0  # Disable profiling to avoid overhead
    )
    return tensorboard_callback



def run_hyperparameter_tuning(input_shape, nfilt, data_parameters, training_parameters, gen, validation_data, run_name="debug", callbacks=None):
    # Create the hypermodel
    hypermodel = SwimStrokeHyperModel(input_shape, nfilt, training_parameters, data_parameters)

    # Check if hypermodel is valid
    if hypermodel is None or not isinstance(hypermodel, HyperModel):
        print("Invalid hypermodel configuration. Skipping tuning for this trial.")
        return None  # Skip this trial
    
    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        objective = kt.Objective('val_single_threshold_objective', direction='max')
    elif training_parameters['swim_style_output']:
        objective=kt.Objective('val_weighted_categorical_accuracy', direction='max')
    else:
        objective=kt.Objective('val_weighted_f1_score', direction='max')



    # Define the Bayesian Optimization tuner
    tuner = BayesianOptimization(
        hypermodel=hypermodel,
        objective=objective,
        max_trials=200,  # Number of trials (adjust based on resources)
        num_initial_points=40,  # Number of random points to start the search
        directory=f'hyperparameter_{run_name}',
        project_name='swim_stroke_model',
        max_consecutive_failed_trials=10 # Increase max_failures

    )

           
    # Set up TensorBoard callback
    log_dir = f"logs/{run_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = get_tensorboard_callback(log_dir)

    # Add TensorBoard callback to the list of callbacks
    if callbacks is None:
        callbacks = []
    callbacks.append(tensorboard_callback)
    # Add debugging callback
    #callbacks.append(DebugCallback())
    # Start the search

    tuner.search(
        gen,
        validation_data=validation_data,
        batch_size=training_parameters['batch_size'],
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
   #  validation_steps=training_parameters['steps_per_epoch'],  # Define a fixed number of validation steps
        callbacks=callbacks
    )

    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model, best_hps

if __name__ == '__main__':
    input_shape = (180, 6)
    nfilt = 32  # Number of filters in the WaveNet model
    data_parameters = {'label_type': 'majority', 'debug': True}

    training_parameters = get_default_training_parameters()
    # Example of class weights (modify it to suit your needs)
    # Generate dummy data for demonstration

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        def gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6)),  # Features
                    {  # Labels
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )

        def val_gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6)),  # Features
                    { 
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )
        # Set up the combined early stopping callback
        callbacks = [CombinedEarlyStopping(
            monitor1='val_stroke_label_output_weighted_f1_score',
            monitor2='val_swim_style_ouput_weighted_categorical_accuracy',
            mode1='max',
            mode2='max',
            patience=5,
            restore_best_weights=True
        )]
    elif training_parameters['swim_style_output']:  # Only swim style output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])

        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_categorical_accuracy', patience=10, restore_best_weights=True, mode='max')]
    else:  # Only stroke label output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=10, restore_best_weights=True, mode='max')]

    best_model = run_hyperparameter_tuning(input_shape, nfilt, data_parameters, training_parameters, gen(), val_gen(), callbacks=callbacks)
    best_model.summary()
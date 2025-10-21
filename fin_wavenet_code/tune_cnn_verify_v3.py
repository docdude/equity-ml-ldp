# tune_cnn_verify.py - Fixed version with proper dimension tracking
import tensorflow as tf
from keras_tuner import HyperModel, RandomSearch, BayesianOptimization
from utils_train import compile_model, weighted_binary_crossentropy_smooth_class, EarlyStoppingLogger, CombinedEarlyStopping, CombinedMetricCallback
import keras_tuner as kt
import pickle
import os
import datetime
import traceback
import numpy as np


def get_default_training_parameters():
    """Get a default set of parameters used to train a cnn model"""
    training_parameters = {'swim_style_lr': 0.0005,
                        'stroke_lr': {
                                'initial_lr': 0.0005,
                                'decay_steps': 1000,
                                'decay_rate': 0.9
                            },
                        'beta_1':          0.9,
                        'beta_2':          0.999,
                        'batch_size':      64,
                        'max_epochs':      48,
                        'steps_per_epoch': 100,
                        'noise_std':       0.01,
                        'mirror_prob':     0.5,
                        'random_rot_deg':  30,
                        'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                        'labels':          [0, 1, 2, 3, 4],
                        'stroke_labels': ['stroke_labels'],
                        'stroke_label_output':      True,
                        'swim_style_output':        True,
                        'output_bias':              None
                        }
    return training_parameters


def create_swim_model_parameters(hp):
    # Reduce max pooling options to avoid dimension issues
    return {
        'filters': [hp.Int(f'swim_filters_{i}', min_value=32, max_value=128, step=16) for i in range(4)],
        'kernel_sizes': [hp.Choice(f'swim_kernel_size_{i}', values=[1, 3, 5]) for i in range(4)],
        'strides': [hp.Choice(f'swim_stride_{i}', values=[0, 1]) for i in range(4)],
        'max_pooling': [hp.Choice(f'swim_max_pooling_{i}', values=[0, 2]) for i in range(4)],  # Reduced from [0,1,2,3]
        'units': [hp.Int(f'swim_dense_units_{i}', min_value=64, max_value=256, step=64) for i in range(1)],
        'activation': [hp.Choice(f'swim_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(5)],  
        'batch_norm': [hp.Boolean(f'swim_batch_norm_{i}') for i in range(5)],
        'drop_out': [hp.Float(f'swim_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(5)],
        'max_norm': [hp.Choice(f'swim_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(5)],
        'l2_reg': [hp.Choice(f'swim_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(5)],
        'labels': [0, 1, 2, 3, 4]
    }


def create_stroke_model_parameters(hp):
    return {
        'filters': [hp.Int(f'stroke_filters_{i}', min_value=16, max_value=128, step=16) for i in range(2)],
        'kernel_sizes': [hp.Choice(f'stroke_kernel_size_{i}', values=[1, 3, 5]) for i in range(2)],
        'strides': [hp.Choice(f'stroke_stride_{i}', values=[0]) for i in range(2)],  # Fixed at 0
        'max_pooling': [hp.Choice(f'stroke_max_pooling_{i}', values=[0]) for i in range(2)],  # No pooling
        'units': [hp.Int('stroke_lstm_units', min_value=16, max_value=128, step=16)],  
        'lstm_dropout': hp.Float('stroke_lstm_dropout', min_value=0.0, max_value=0.5, step=0.1),  
        'lstm_recurrent_dropout': hp.Float('stroke_lstm_recurrent_dropout', min_value=0.0, max_value=0.5, step=0.1),  
        'lstm_l2_reg': hp.Choice('stroke_lstm_l2_reg', values=[0.0, 1e-4, 1e-3]),  
        'lstm_max_norm': hp.Choice('stroke_lstm_max_norm', values=[0.0, 0.1, 0.5, 1.0, 4.0]),  
        'activation': [hp.Choice(f'stroke_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(2)],  
        'batch_norm': [hp.Boolean(f'stroke_batch_norm_{i}') for i in range(3)],
        'drop_out': [hp.Float(f'stroke_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(3)],
        'max_norm': [hp.Choice(f'stroke_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(3)],
        'l2_reg': [hp.Choice(f'stroke_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(3)],
        'final_dropout': hp.Float('stroke_final_dropout', min_value=0.0, max_value=0.5, step=0.1),  
        'stroke_labels': ['stroke_labels']
    }


def create_loss_parameters(hp):
    return {
        'alpha': [hp.Float(f'alpha', min_value=0.75, max_value=0.95, step=0.05)],
        'gamma': [hp.Float(f'gamma', min_value=1.0, max_value=2.0, step=0.05)],  
        'target_height': [hp.Float(f'target_height', min_value=0.80, max_value=0.90, step=0.05)],
        'peak_weight': [hp.Float(f'peak_weight', min_value=10.0, max_value=15.0, step=1.0)],
        'peak_threshold': [hp.Float(f'peak_threshold', min_value=0.55, max_value=0.70, step=0.05)]
    }


def validate_model_dimensions_detailed(swim_model_parameters=None, stroke_model_parameters=None, 
                                      input_shape=(180, 6, 1), verbose=False):
    """
    Thoroughly validate model dimensions by simulating the exact layer-by-layer transformations.
    
    :param swim_model_parameters: Dictionary containing hyperparameters for the swim model.
    :param stroke_model_parameters: Dictionary containing hyperparameters for the stroke model.
    :param input_shape: Tuple representing the shape of the input data (height, width, channels).
    :param verbose: Print dimension changes at each layer
    :return: Tuple (is_valid, error_message, dimension_trace)
    """
    height, width, channels = input_shape
    dimension_trace = []
    
    # Validate swim model parameters
    if swim_model_parameters is not None:
        temp_height, temp_width = height, width
        dimension_trace.append(f"Swim Model Input: ({temp_height}, {temp_width}, {channels})")
        
        for i in range(len(swim_model_parameters['filters'])):
            layer_desc = f"Swim Layer {i}:"
            kernel_size = swim_model_parameters['kernel_sizes'][i]
            stride = swim_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            
            max_pooling = swim_model_parameters['max_pooling'][i]
            
            # Conv2D with 'same' padding
            # Output dimensions with 'same' padding: ceil(input / stride)
            conv_output_height = int(np.ceil(temp_height / stride))
            conv_output_width = int(np.ceil(temp_width / stride))
            
            layer_desc += f" Conv(k={kernel_size}, s={stride}) -> ({conv_output_height}, {conv_output_width})"
            
            # Check if dimensions are still positive after conv
            if conv_output_height <= 0 or conv_output_width <= 0:
                error_msg = f"Swim model layer {i}: After Conv2D, dimensions become non-positive ({conv_output_height}, {conv_output_width})"
                return False, error_msg, dimension_trace
            
            temp_height = conv_output_height
            temp_width = conv_output_width
            
            # MaxPooling2D (default padding is 'valid')
            if max_pooling > 0:
                # Check if pooling size exceeds current dimensions
                if max_pooling > temp_height:
                    error_msg = f"Swim model layer {i}: Max pooling size {max_pooling} exceeds height {temp_height}"
                    dimension_trace.append(layer_desc + f" -> ERROR: pool size {max_pooling} > height {temp_height}")
                    return False, error_msg, dimension_trace
                
                # MaxPooling2D with 'valid' padding: floor((input - pool_size) / pool_size) + 1
                # Since the pooling is only on height dimension (pool_size, 1)
                pool_output_height = (temp_height - max_pooling) // max_pooling + 1
                pool_output_width = temp_width  # Width unchanged
                
                layer_desc += f", MaxPool({max_pooling}) -> ({pool_output_height}, {pool_output_width})"
                
                # Check if dimensions are still positive after pooling
                if pool_output_height <= 0:
                    error_msg = f"Swim model layer {i}: After MaxPooling, height becomes non-positive ({pool_output_height})"
                    dimension_trace.append(layer_desc + " -> ERROR: non-positive height")
                    return False, error_msg, dimension_trace
                
                temp_height = pool_output_height
                temp_width = pool_output_width
            
            dimension_trace.append(layer_desc)
            
            if verbose:
                print(layer_desc)
    
    # Validate stroke model parameters
    if stroke_model_parameters is not None:
        temp_height, temp_width = height, width
        dimension_trace.append(f"Stroke Model Input: ({temp_height}, {temp_width}, {channels})")
        
        for i in range(len(stroke_model_parameters['filters'])):
            layer_desc = f"Stroke Layer {i}:"
            kernel_size = stroke_model_parameters['kernel_sizes'][i]
            stride = stroke_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            max_pooling = stroke_model_parameters['max_pooling'][i]
            
            # Conv2D with 'same' padding
            conv_output_height = int(np.ceil(temp_height / stride))
            conv_output_width = int(np.ceil(temp_width / stride))
            
            layer_desc += f" Conv(k={kernel_size}, s={stride}) -> ({conv_output_height}, {conv_output_width})"
            
            # Check if dimensions are still positive after conv
            if conv_output_height <= 0 or conv_output_width <= 0:
                error_msg = f"Stroke model layer {i}: After Conv2D, dimensions become non-positive ({conv_output_height}, {conv_output_width})"
                return False, error_msg, dimension_trace
            
            temp_height = conv_output_height
            temp_width = conv_output_width
            
            # MaxPooling2D if specified
            if max_pooling > 0:
                if max_pooling > temp_height:
                    error_msg = f"Stroke model layer {i}: Max pooling size {max_pooling} exceeds height {temp_height}"
                    dimension_trace.append(layer_desc + f" -> ERROR: pool size {max_pooling} > height {temp_height}")
                    return False, error_msg, dimension_trace
                
                pool_output_height = (temp_height - max_pooling) // max_pooling + 1
                pool_output_width = temp_width
                
                layer_desc += f", MaxPool({max_pooling}) -> ({pool_output_height}, {pool_output_width})"
                
                if pool_output_height <= 0:
                    error_msg = f"Stroke model layer {i}: After MaxPooling, height becomes non-positive ({pool_output_height})"
                    dimension_trace.append(layer_desc + " -> ERROR: non-positive height")
                    return False, error_msg, dimension_trace
                
                temp_height = pool_output_height
                temp_width = pool_output_width
            
            dimension_trace.append(layer_desc)
            
            if verbose:
                print(layer_desc)
        
        # Check final dimensions for LSTM
        if temp_height <= 0:
            error_msg = f"Stroke model: Final temporal dimension is non-positive ({temp_height})"
            return False, error_msg, dimension_trace
    
    return True, "Valid configuration", dimension_trace


class SwimStrokeHyperModel(HyperModel):
    def __init__(self, input_shape, training_parameters, data_parameters):
        self.input_shape = input_shape
        self.training_parameters = training_parameters
        self.data_parameters = data_parameters

    def build(self, hp):
        try:
            swim_model_parameters = (
                create_swim_model_parameters(hp)
                if self.training_parameters['swim_style_output']
                else None
            )
            stroke_model_parameters = (
                create_stroke_model_parameters(hp)
                if self.training_parameters['stroke_label_output']
                else None
            )

            loss_parameters = create_loss_parameters(hp)

            # Add learning rate tuning
            swim_style_lr = hp.Float('swim_style_lr', min_value=1e-5, max_value=1e-2, sampling='log')
            stroke_lr = hp.Float('stroke_lr', min_value=1e-5, max_value=1e-2, sampling='log')
            self.training_parameters['swim_style_lr'] = swim_style_lr
            self.training_parameters['stroke_lr'] = stroke_lr

            # Detailed validation with dimension tracking
            is_valid, error_msg, dimension_trace = validate_model_dimensions_detailed(
                swim_model_parameters, stroke_model_parameters, self.input_shape
            )
            
            if not is_valid:
                print(f"Invalid configuration detected: {error_msg}")
                for trace in dimension_trace[-3:]:  # Show last few dimension changes
                    print(f"  {trace}")
                # Return None to signal invalid configuration to tuner
                return None
            
            # Additional validations
            is_valid_loss = validate_loss_parameters(loss_parameters)
            is_valid_training = validate_training_parameters(self.training_parameters)
            
            if not (is_valid_loss and is_valid_training):
                return None

            # Build the CNN model
            model = cnn_model(
                input_shape=self.input_shape,
                swim_model_parameters=swim_model_parameters,
                stroke_model_parameters=stroke_model_parameters,
                training_parameters=self.training_parameters,
            )
            
            if model is None:
                return None
            
            # Compile the model
            model = compile_model(
                data_parameters=self.data_parameters,
                model=model,
                training_parameters=self.training_parameters,
                loss_parameters=loss_parameters,
            )

            return model
        
        except (ValueError, InvalidHyperparameterConfiguration) as e:
            print(f"Error in model building: {e}. Skipping this trial.")
            # Return a dummy model that will fail quickly
            return create_dummy_model(self.input_shape, self.training_parameters)
        except Exception as e:
            print(f"Unexpected error in model building: {e}. Skipping this trial.")
            traceback.print_exc()
            return create_dummy_model(self.input_shape, self.training_parameters)


def validate_loss_parameters(loss_parameters):
    """Validate loss parameters to ensure they are within reasonable ranges."""
    if loss_parameters is None:
        return True
        
    try:
        if 'alpha' in loss_parameters:
            alpha = loss_parameters['alpha'][0] if isinstance(loss_parameters['alpha'], list) else loss_parameters['alpha']
            if not (0.0 <= alpha <= 1.0):
                return False
        
        if 'gamma' in loss_parameters:
            gamma = loss_parameters['gamma'][0] if isinstance(loss_parameters['gamma'], list) else loss_parameters['gamma']
            if gamma < 0 or gamma > 5.0:
                return False
        
        if 'target_height' in loss_parameters:
            target_height = loss_parameters['target_height'][0] if isinstance(loss_parameters['target_height'], list) else loss_parameters['target_height']
            if not (0.0 <= target_height <= 1.0):
                return False
        
        if 'peak_weight' in loss_parameters:
            peak_weight = loss_parameters['peak_weight'][0] if isinstance(loss_parameters['peak_weight'], list) else loss_parameters['peak_weight']
            if peak_weight <= 0 or peak_weight > 100:
                return False
        
        if 'peak_threshold' in loss_parameters:
            peak_threshold = loss_parameters['peak_threshold'][0] if isinstance(loss_parameters['peak_threshold'], list) else loss_parameters['peak_threshold']
            if not (0.0 <= peak_threshold <= 1.0):
                return False
        
        return True
    except Exception:
        return False

def create_dummy_model(input_shape, training_parameters):
    """
    Create a minimal dummy model for failed trials.
    This model will compile but perform poorly, causing the tuner to skip it.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    
    outputs = []
    if training_parameters.get('swim_style_output'):
        swim_output = tf.keras.layers.Dense(5, activation='softmax', name='swim_style_output')(x)
        outputs.append(swim_output)
    
    if training_parameters.get('stroke_label_output'):
        stroke_output = tf.keras.layers.Dense(1, activation='sigmoid', name='stroke_label_output')(x)
        stroke_output = tf.keras.layers.Reshape((1, 1), name='stroke_reshape')(stroke_output)
        outputs.append(stroke_output)
    
    if len(outputs) == 0:
        outputs = tf.keras.layers.Dense(1)(x)
    elif len(outputs) == 1:
        outputs = outputs[0]
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Copy the model building functions from tune_cnn.py
def cnn_model(input_shape, swim_model_parameters=None, stroke_model_parameters=None, training_parameters=None):
    inputs = tf.keras.Input(shape=input_shape)

    # Build swim style branch if swim_model_parameters are provided
    swim_style_output = None
    if swim_model_parameters is not None and training_parameters['swim_style_output']:
        swim_style_output = swim_style_model(inputs, swim_model_parameters, output_bias=None)

    # Build stroke branch if stroke_model_parameters are provided
    stroke_label_output = None
    if stroke_model_parameters is not None and training_parameters['stroke_label_output']:
        stroke_label_output = stroke_model(inputs, stroke_model_parameters, output_bias=training_parameters['output_bias'])

    # Combine outputs based on the branches enabled
    if swim_style_output is not None and stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])
    elif swim_style_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=swim_style_output)
    elif stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=stroke_label_output)
    else:
        raise ValueError("No outputs selected for the model.")

    return model


# Copy swim_style_model and stroke_model functions from tune_cnn.py
def swim_style_model(inputs, swim_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for swim style classification
    :param input_shape: The shape of the input data
    :param swim_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    num_cl = len(swim_model_parameters['filters'])
    num_fcl = len(swim_model_parameters['units'])
    cnt_layer = 0

    swim_style_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Main convolutional layers (for swim style)
    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        strides = swim_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        swim_style_branch = tf.keras.layers.Conv2D(
            filters=swim_model_parameters['filters'][i],
            kernel_size=(swim_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            padding='same',  # Use 'same' padding
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
            #kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337), # Add kernel initializer original implementation
            bias_initializer="zeros",
            name=f'swim_style_conv_{i}'
        )(swim_style_branch)

        if swim_model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(
                name=f'swim_style_bn_{i}'
            )(swim_style_branch)
        # Handle different activation functions
        activation_function = swim_model_parameters['activation'][cnt_layer]
        if activation_function == 'leaky_relu':
            swim_style_branch = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'swim_style_activation_{i}')(swim_style_branch)
        else:
            swim_style_branch = tf.keras.layers.Activation(activation_function, name=f'swim_style_activation_{i}')(swim_style_branch)
        max_pooling = swim_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            swim_style_branch = tf.keras.layers.MaxPooling2D(
                #1,
                (max_pooling, 1),
                name=f'swim_style_pool_{i}'
            )(swim_style_branch)
        if swim_model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(
                swim_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'swim_style_dropout_{i}'
            )(swim_style_branch)
        cnt_layer += 1

    # Swim Style Branch
    swim_style_branch = tf.keras.layers.Flatten(
        name='swim_style_flatten'
    )(swim_style_branch)
    
    for i in range(num_fcl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        swim_style_branch = tf.keras.layers.Dense(
            units=swim_model_parameters['units'][i],
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
            bias_initializer='zeros',
            name=f'swim_style_dense_{i}'
        )(swim_style_branch)
        
        if swim_model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(
                name=f'swim_style_dense_bn_{i}'
            )(swim_style_branch)
        swim_style_branch = tf.keras.layers.Activation(
            swim_model_parameters['activation'][cnt_layer],
            name=f'swim_style_dense_activation_{i}'
        )(swim_style_branch)
        if swim_model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(
                swim_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'swim_style_dense_dropout_{i}'
            )(swim_style_branch)
        cnt_layer += 1

    # Swim Style Output
    swim_style_output = tf.keras.layers.Dense(
        len(swim_model_parameters['labels']),
        activation="softmax",
        #kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),

        name="swim_style_output"
    )(swim_style_branch)

    return swim_style_output

# Define the stroke model (copied from cnn_vanilla_dual.py and modified for tuning)
def stroke_model(inputs, stroke_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for stroke detection
    :param input_shape: The shape of the input data
    :param stroke_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    # Stroke Detection Branch
    num_stroke_cl = len(stroke_model_parameters['filters'])
    cnt_layer = 0
    stroke_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Convolutional layers for stroke detection
    for i in range(num_stroke_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][cnt_layer])
            if stroke_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][cnt_layer])
            if stroke_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )

        kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337)

        strides = stroke_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        stroke_branch = tf.keras.layers.Conv2D(
            filters=stroke_model_parameters['filters'][i],
            kernel_size=(stroke_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            padding='same',
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            name=f'stroke_conv_{i}'
        )(stroke_branch)

        if stroke_model_parameters['batch_norm'][cnt_layer]:
            stroke_branch = tf.keras.layers.BatchNormalization(
                name=f'stroke_bn_{i}'
            )(stroke_branch)
        stroke_branch = tf.keras.layers.Activation(
            stroke_model_parameters['activation'][cnt_layer],
            name=f'stroke_activation_{i}'
        )(stroke_branch)
        max_pooling = stroke_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            stroke_branch = tf.keras.layers.MaxPooling2D(
                #1,
                (max_pooling, 1),
                name=f'stroke_pool_{i}'
            )(stroke_branch)
        if stroke_model_parameters['drop_out'][cnt_layer] is not None:
            stroke_branch = tf.keras.layers.Dropout(
                stroke_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'stroke_dropout_{i}'
            )(stroke_branch)
        
        # If this is the first conv layer, create attention mechanism
        if i == 0:
            attention = stroke_branch
        cnt_layer += 1

    # Apply global pooling to the attention tensor
    attention = tf.keras.layers.GlobalAveragePooling2D(name='stroke_attention_global_pooling')(attention)

    # Expand dimensions of the attention tensor to match the feature tensor
    attention = tf.keras.layers.Dense(
        units=stroke_branch.shape[-1],  # Match the number of filters in the feature tensor
        activation='sigmoid',  # Use sigmoid to scale attention values between 0 and 1
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
        name='stroke_attention_dense'
    )(attention)
    attention = tf.keras.layers.Reshape((1, 1, stroke_branch.shape[-1]), name='stroke_attention_reshape')(attention)

    # Multiply attention with features
    stroke_branch = tf.keras.layers.Multiply(name='stroke_multiply')([stroke_branch, attention])

    
    # Reshape to (batch, time, features)
    temporal_dim = stroke_branch.shape[1]
    stroke_branch = tf.keras.layers.Reshape(
        (temporal_dim, -1),
        name='stroke_reshape'
    )(stroke_branch)
    
    # LSTM layer
    stroke_branch = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=stroke_model_parameters['units'][0],  # Tunable LSTM units
            return_sequences=True,
            dropout=stroke_model_parameters['lstm_dropout'],  # Tunable LSTM dropout
            recurrent_dropout=stroke_model_parameters['lstm_recurrent_dropout'],  # Tunable recurrent dropout
            recurrent_initializer='orthogonal',  # Good for LSTM
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
            kernel_constraint=tf.keras.constraints.max_norm(stroke_model_parameters['lstm_max_norm'])  # Tunable max norm
                if stroke_model_parameters['lstm_max_norm'] != 0
                else None,
            kernel_regularizer=tf.keras.regularizers.l2(stroke_model_parameters['lstm_l2_reg'])  # Tunable L2 regularization
                if stroke_model_parameters['lstm_l2_reg'] != 0
                else None,
            bias_initializer='zeros',
            name='stroke_lstm'
        ),
        merge_mode='concat',
        name='stroke_bilstm'
    )(stroke_branch)

    if stroke_model_parameters['final_dropout'] > 0.0:
        stroke_branch = tf.keras.layers.Dropout(
            stroke_model_parameters['final_dropout'],
            name='stroke_final_dropout'
        )(stroke_branch)

    stroke_branch = tf.keras.layers.LayerNormalization(
        name='stroke_layer_norm'
    )(stroke_branch)

    # Stroke detection output
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(stroke_model_parameters['stroke_labels']),
            activation="sigmoid",
            bias_initializer=output_bias,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
            name='stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_branch)

    return stroke_label_output
def validate_training_parameters(training_parameters):
    """Validate training parameters."""
    try:
        if 'swim_style_lr' in training_parameters:
            if training_parameters['swim_style_lr'] <= 0 or training_parameters['swim_style_lr'] > 1:
                return False
        
        if 'stroke_lr' in training_parameters:
            if isinstance(training_parameters['stroke_lr'], dict):
                if training_parameters['stroke_lr'].get('initial_lr', 0) <= 0:
                    return False
            elif training_parameters['stroke_lr'] <= 0 or training_parameters['stroke_lr'] > 1:
                return False
        
        if 'batch_size' in training_parameters:
            if training_parameters['batch_size'] <= 0:
                return False
        
        if 'beta_1' in training_parameters:
            if not (0.0 <= training_parameters['beta_1'] < 1.0):
                return False
        
        if 'beta_2' in training_parameters:
            if not (0.0 <= training_parameters['beta_2'] < 1.0):
                return False
        
        return True
    except Exception:
        return False



def test_specific_configuration(input_shape=(180, 6, 1)):
    """Test the specific configuration that was failing."""
    # The failing configuration from your error
    swim_params = {
        'filters': [64, 112, 128, 48],
        'kernel_sizes': [3, 3, 5, 5],
        'strides': [2, 0, 2, 0],  # 0 means stride=1
        'max_pooling': [3, 3, 2, 3],
        'units': [192],
        'activation': ['leaky_relu'] * 5,
        'batch_norm': [False, False, False, False, True],
        'drop_out': [0.3, 0, 0.2, 0, 0.1],
        'max_norm': [0.5, 0.5, 0.5, 1, 1],
        'l2_reg': [0, 0.001, 0.001, 0, 0.001],
        'labels': [0, 1, 2, 3, 4]
    }
    
    print("Testing problematic configuration:")
    print("-" * 50)
    is_valid, error_msg, dimension_trace = validate_model_dimensions_detailed(
        swim_params, None, input_shape, verbose=True
    )
    
    print("\nDimension Trace:")
    for trace in dimension_trace:
        print(trace)
    
    if not is_valid:
        print(f"\n❌ Configuration is INVALID: {error_msg}")
    else:
        print("\n✅ Configuration is VALID")
    
    return is_valid


if __name__ == '__main__':
    # Test the specific failing configuration
    print("Testing the configuration that caused the error:")
    test_specific_configuration()
    
    print("\n" + "="*80)
    print("Running general validation test...")
    
    input_shape = (180, 6, 1)
    data_parameters = {'label_type': 'majority', 'debug': False}
    training_parameters = get_default_training_parameters()
    
    # Create a simple test
    hp = kt.HyperParameters()
    hp.Fixed('swim_filters_0', 64)
    hp.Fixed('swim_filters_1', 64)
    hp.Fixed('swim_filters_2', 64) 
    hp.Fixed('swim_filters_3', 64)
    hp.Fixed('swim_kernel_size_0', 3)
    hp.Fixed('swim_kernel_size_1', 3)
    hp.Fixed('swim_kernel_size_2', 3)
    hp.Fixed('swim_kernel_size_3', 3)
    hp.Fixed('swim_stride_0', 1)
    hp.Fixed('swim_stride_1', 0)
    hp.Fixed('swim_stride_2', 1)
    hp.Fixed('swim_stride_3', 0)
    hp.Fixed('swim_max_pooling_0', 2)
    hp.Fixed('swim_max_pooling_1', 2)
    hp.Fixed('swim_max_pooling_2', 2)
    hp.Fixed('swim_max_pooling_3', 0)  # No pooling in last layer to avoid dimension issues
    
    # Test with safer configuration
    print("\nTesting safer configuration:")
    hypermodel = SwimStrokeHyperModel(input_shape, training_parameters, data_parameters)
    result = hypermodel.build(hp)
    
    if result is not None:
        print("✅ Model built successfully with safer parameters")
    else:
        print("❌ Model build failed even with safer parameters")

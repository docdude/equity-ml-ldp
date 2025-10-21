# tune_cnn_validate_all.py - Comprehensive validation for all hyperparameter trials
import tensorflow as tf
from keras_tuner import HyperModel, BayesianOptimization
import keras_tuner as kt
import numpy as np
import os
import datetime
import shutil
from tune_cnn import (
    create_swim_model_parameters, 
    create_stroke_model_parameters, 
    create_loss_parameters,
    get_default_training_parameters,
    SwimStrokeHyperModel
)


def validate_model_dimensions_comprehensive(swim_model_parameters=None, stroke_model_parameters=None, 
                                           input_shape=(180, 6, 1), verbose=False):
    """
    Comprehensively validate model dimensions by simulating exact layer transformations.
    """
    height, width, channels = input_shape
    errors = []
    
    # Validate swim model parameters
    if swim_model_parameters is not None:
        temp_height, temp_width = height, width
        
        for i in range(len(swim_model_parameters['filters'])):
            kernel_size = swim_model_parameters['kernel_sizes'][i]
            stride = swim_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            max_pooling = swim_model_parameters['max_pooling'][i]
            
            # Conv2D with 'same' padding: output = ceil(input / stride)
            conv_output_height = int(np.ceil(temp_height / stride))
            conv_output_width = int(np.ceil(temp_width / stride))
            
            if conv_output_height <= 0 or conv_output_width <= 0:
                errors.append(f"Swim layer {i} Conv: dimensions become non-positive ({conv_output_height}, {conv_output_width})")
                return False, errors
            
            temp_height = conv_output_height
            temp_width = conv_output_width
            
            # MaxPooling2D with 'valid' padding
            if max_pooling > 0:
                if max_pooling > temp_height:
                    errors.append(f"Swim layer {i} MaxPool: pool size {max_pooling} > height {temp_height}")
                    return False, errors
                
                pool_output_height = (temp_height - max_pooling) // max_pooling + 1
                pool_output_width = temp_width
                
                if pool_output_height <= 0:
                    errors.append(f"Swim layer {i} MaxPool: height becomes non-positive ({pool_output_height})")
                    return False, errors
                
                temp_height = pool_output_height
                temp_width = pool_output_width
        
        if verbose:
            print(f"  Swim final dimensions: ({temp_height}, {temp_width})")
    
    # Validate stroke model parameters
    if stroke_model_parameters is not None:
        temp_height, temp_width = height, width
        
        for i in range(len(stroke_model_parameters['filters'])):
            kernel_size = stroke_model_parameters['kernel_sizes'][i]
            stride = stroke_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            max_pooling = stroke_model_parameters['max_pooling'][i]
            
            # Conv2D with 'same' padding
            conv_output_height = int(np.ceil(temp_height / stride))
            conv_output_width = int(np.ceil(temp_width / stride))
            
            if conv_output_height <= 0 or conv_output_width <= 0:
                errors.append(f"Stroke layer {i} Conv: dimensions become non-positive ({conv_output_height}, {conv_output_width})")
                return False, errors
            
            temp_height = conv_output_height
            temp_width = conv_output_width
            
            # MaxPooling2D if specified
            if max_pooling > 0:
                if max_pooling > temp_height:
                    errors.append(f"Stroke layer {i} MaxPool: pool size {max_pooling} > height {temp_height}")
                    return False, errors
                
                pool_output_height = (temp_height - max_pooling) // max_pooling + 1
                pool_output_width = temp_width
                
                if pool_output_height <= 0:
                    errors.append(f"Stroke layer {i} MaxPool: height becomes non-positive ({pool_output_height})")
                    return False, errors
                
                temp_height = pool_output_height
                temp_width = pool_output_width
        
        if temp_height <= 0:
            errors.append(f"Stroke model: Final temporal dimension is non-positive ({temp_height})")
            return False, errors
        
        if verbose:
            print(f"  Stroke final dimensions: ({temp_height}, {temp_width})")
    
    return True, []


def extract_hyperparameters_for_validation(hp_values):
    """Extract the relevant hyperparameters for dimension validation."""
    swim_params = None
    stroke_params = None
    
    # Check if swim parameters exist
    if 'swim_filters_0' in hp_values:
        swim_params = {
            'filters': [hp_values.get(f'swim_filters_{i}', 64) for i in range(4)],
            'kernel_sizes': [hp_values.get(f'swim_kernel_size_{i}', 3) for i in range(4)],
            'strides': [hp_values.get(f'swim_stride_{i}', 0) for i in range(4)],
            'max_pooling': [hp_values.get(f'swim_max_pooling_{i}', 0) for i in range(4)],
            'units': [hp_values.get(f'swim_dense_units_{i}', 128) for i in range(1)],
            'activation': [hp_values.get(f'swim_activation_{i}', 'relu') for i in range(5)],
            'batch_norm': [hp_values.get(f'swim_batch_norm_{i}', False) for i in range(5)],
            'drop_out': [hp_values.get(f'swim_dropout_{i}', 0.0) for i in range(5)],
            'max_norm': [hp_values.get(f'swim_max_norm_{i}', 0.0) for i in range(5)],
            'l2_reg': [hp_values.get(f'swim_l2_reg_{i}', 0.0) for i in range(5)],
            'labels': [0, 1, 2, 3, 4]
        }
    
    # Check if stroke parameters exist
    if 'stroke_filters_0' in hp_values:
        stroke_params = {
            'filters': [hp_values.get(f'stroke_filters_{i}', 32) for i in range(2)],
            'kernel_sizes': [hp_values.get(f'stroke_kernel_size_{i}', 3) for i in range(2)],
            'strides': [hp_values.get(f'stroke_stride_{i}', 0) for i in range(2)],
            'max_pooling': [hp_values.get(f'stroke_max_pooling_{i}', 0) for i in range(2)],
            'units': [hp_values.get('stroke_lstm_units', 32)],
            'lstm_dropout': hp_values.get('stroke_lstm_dropout', 0.0),
            'lstm_recurrent_dropout': hp_values.get('stroke_lstm_recurrent_dropout', 0.0),
            'lstm_l2_reg': hp_values.get('stroke_lstm_l2_reg', 0.0),
            'lstm_max_norm': hp_values.get('stroke_lstm_max_norm', 0.0),
            'activation': [hp_values.get(f'stroke_activation_{i}', 'relu') for i in range(2)],
            'batch_norm': [hp_values.get(f'stroke_batch_norm_{i}', False) for i in range(3)],
            'drop_out': [hp_values.get(f'stroke_dropout_{i}', 0.0) for i in range(3)],
            'max_norm': [hp_values.get(f'stroke_max_norm_{i}', 0.0) for i in range(3)],
            'l2_reg': [hp_values.get(f'stroke_l2_reg_{i}', 0.0) for i in range(3)],
            'final_dropout': hp_values.get('stroke_final_dropout', 0.0),
            'stroke_labels': ['stroke_labels']
        }
    
    return swim_params, stroke_params


def validate_all_trials(input_shape, training_parameters, data_parameters, num_trials=200):
    """
    Generate and validate all hyperparameter combinations for the specified number of trials.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING {num_trials} HYPERPARAMETER TRIALS")
    print(f"{'='*80}\n")
    
    # Create the hypermodel
    hypermodel = SwimStrokeHyperModel(input_shape, training_parameters, data_parameters)
    
    # Determine objective and metric name based on outputs
    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        objective = kt.Objective('val_combined_metric', direction='max')
        metric_name = 'val_combined_metric'
        dummy_value = 0.0  # Use 0 for metrics we want to maximize
    elif training_parameters['swim_style_output']:
        objective = kt.Objective('val_weighted_categorical_accuracy', direction='max')
        metric_name = 'val_weighted_categorical_accuracy'
        dummy_value = 0.0
    else:
        objective = kt.Objective('val_weighted_f1_score', direction='max')
        metric_name = 'val_weighted_f1_score'
        dummy_value = 0.0
    
    print(f"Objective: {objective.name} (direction: {objective.direction})")
    print(f"Expected metric: {metric_name}\n")
    
    # Create a temporary directory for the tuner
    temp_dir = f'temp_validation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Create the Bayesian Optimization tuner
    tuner = BayesianOptimization(
        hypermodel=hypermodel,
        objective=objective,
        max_trials=num_trials,
        num_initial_points=40,
        directory=temp_dir,
        project_name='validation_test',
        overwrite=True
    )
    
    # Generate all trial configurations
    invalid_trials = []
    valid_trials = []
    problematic_configs = {}
    
    print("Generating and validating trial configurations...")
    print("-" * 50)
    
    for trial_id in range(num_trials):
        try:
            # Generate a trial using the oracle
            trial = tuner.oracle.create_trial(f"trial_{trial_id:04d}")
            
            if trial is None:
                print(f"⚠️ Trial {trial_id:3d}: Could not create trial (oracle may have exhausted search space)")
                continue
                
            hp_values = trial.hyperparameters.values
            
            # Extract parameters for validation
            swim_params, stroke_params = extract_hyperparameters_for_validation(hp_values)
            
            # Validate dimensions
            is_valid, errors = validate_model_dimensions_comprehensive(
                swim_params, stroke_params, input_shape, verbose=False
            )
            
            if is_valid:
                valid_trials.append(trial_id)
                if trial_id % 20 == 0:
                    print(f"✅ Trial {trial_id:3d}: VALID")
            else:
                invalid_trials.append(trial_id)
                problematic_configs[trial_id] = {
                    'errors': errors,
                    'config': {
                        'swim_strides': [swim_params['strides'][i] for i in range(4)] if swim_params else None,
                        'swim_max_pooling': [swim_params['max_pooling'][i] for i in range(4)] if swim_params else None,
                        'swim_kernel_sizes': [swim_params['kernel_sizes'][i] for i in range(4)] if swim_params else None,
                        'stroke_strides': [stroke_params['strides'][i] for i in range(2)] if stroke_params else None,
                        'stroke_max_pooling': [stroke_params['max_pooling'][i] for i in range(2)] if stroke_params else None,
                    }
                }
                print(f"❌ Trial {trial_id:3d}: INVALID - {errors[0]}")
            
            # Update oracle with the correct metric name
            metrics = {metric_name: dummy_value}
            tuner.oracle.update_trial(trial.trial_id, metrics)
            tuner.oracle.end_trial(trial)
            
        except Exception as e:
            print(f"⚠️ Trial {trial_id:3d}: Error generating configuration - {str(e)}")
            invalid_trials.append(trial_id)
    
    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total trials validated: {num_trials}")
    print(f"✅ Valid configurations: {len(valid_trials)} ({len(valid_trials)/num_trials*100:.1f}%)")
    print(f"❌ Invalid configurations: {len(invalid_trials)} ({len(invalid_trials)/num_trials*100:.1f}%)")
    
    if invalid_trials:
        print(f"\n{'='*80}")
        print("PROBLEMATIC PATTERNS")
        print(f"{'='*80}")
        
        # Analyze common failure patterns
        failure_patterns = {}
        for trial_id, info in problematic_configs.items():
            error_type = info['errors'][0].split(':')[0] if info['errors'] else "Unknown"
            if error_type not in failure_patterns:
                failure_patterns[error_type] = []
            failure_patterns[error_type].append(info['config'])
        
        for error_type, configs in failure_patterns.items():
            print(f"\n{error_type}: {len(configs)} occurrences")
            if configs and len(configs) <= 3:
                for i, config in enumerate(configs[:3]):
                    print(f"  Example {i+1}:")
                    if config['swim_strides']:
                        print(f"    Swim strides: {config['swim_strides']}")
                        print(f"    Swim pooling: {config['swim_max_pooling']}")
    
    # Provide recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if len(invalid_trials) > num_trials * 0.1:  # More than 10% invalid
        print("⚠️ High proportion of invalid configurations detected!")
        print("Consider adjusting hyperparameter ranges in tune_cnn.py:")
        print("\nSuggested changes to create_swim_model_parameters():")
        print("  'max_pooling': [")
        print("      hp.Choice('swim_max_pooling_0', values=[0, 2]),  # Reduced from [0, 1, 2, 3]")
        print("      hp.Choice('swim_max_pooling_1', values=[0, 2]),")
        print("      hp.Choice('swim_max_pooling_2', values=[0]),     # No pooling in layer 2")
        print("      hp.Choice('swim_max_pooling_3', values=[0]),     # No pooling in layer 3")
        print("  ],")
        print("\n  'strides': [")
        print("      hp.Choice('swim_stride_0', values=[0, 1]),")
        print("      hp.Choice('swim_stride_1', values=[0]),         # No stride in layer 1")
        print("      hp.Choice('swim_stride_2', values=[0, 1]),")
        print("      hp.Choice('swim_stride_3', values=[0]),         # No stride in layer 3")
        print("  ],")
    else:
        print("✅ Acceptable proportion of valid configurations")
        print("The tuner should be able to work around invalid configurations")
    
    return valid_trials, invalid_trials, problematic_configs


def fix_hyperparameter_ranges():
    """
    Return fixed hyperparameter ranges that avoid dimension issues.
    """
    print(f"\n{'='*80}")
    print("SUGGESTED SAFER HYPERPARAMETER RANGES")
    print(f"{'='*80}\n")
    
    safer_ranges = """
def create_swim_model_parameters(hp):
    return {
        'filters': [hp.Int(f'swim_filters_{i}', min_value=32, max_value=128, step=16) for i in range(4)],
        'kernel_sizes': [hp.Choice(f'swim_kernel_size_{i}', values=[1, 3, 5]) for i in range(4)],
        'strides': [hp.Choice(f'swim_stride_{i}', values=[0, 1]) for i in range(4)],
        # CHANGED: Reduced max pooling options
        'max_pooling': [
            hp.Choice('swim_max_pooling_0', values=[0, 2]),  # Layer 0
            hp.Choice('swim_max_pooling_1', values=[0, 2]),  # Layer 1
            hp.Choice('swim_max_pooling_2', values=[0, 2]),  # Layer 2
            hp.Choice('swim_max_pooling_3', values=[0]),     # Layer 3 - no pooling
        ],
        'units': [hp.Int(f'swim_dense_units_{i}', min_value=64, max_value=256, step=64) for i in range(1)],
        'activation': [hp.Choice(f'swim_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(5)],
        'batch_norm': [hp.Boolean(f'swim_batch_norm_{i}') for i in range(5)],
        'drop_out': [hp.Float(f'swim_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(5)],
        'max_norm': [hp.Choice(f'swim_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(5)],
        'l2_reg': [hp.Choice(f'swim_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(5)],
        'labels': [0, 1, 2, 3, 4]
    }
"""
    print(safer_ranges)
    
    return safer_ranges


if __name__ == '__main__':
    # Setup
    input_shape = (180, 6, 1)
    data_parameters = {'label_type': 'majority', 'debug': False}
    training_parameters = get_default_training_parameters()
    
    # Run comprehensive validation
    valid_trials, invalid_trials, problematic_configs = validate_all_trials(
        input_shape, training_parameters, data_parameters, num_trials=200
    )
    
    # Show suggested fixes
    if invalid_trials:
        fix_hyperparameter_ranges()
        
        print("\n" + "="*80)
        print("HOW TO FIX:")
        print("="*80)
        print("\n1. Update tune_cnn.py with the safer hyperparameter ranges above")
        print("2. Particularly important: Limit max_pooling in later layers")
        print("3. Consider using different pooling sizes for each layer")
        print("4. The last layer should have no pooling (value=[0])")

"""
Optimized Training Configurations for Financial Time Series
Based on empirical results and best practices
"""

# ==============================================================================
# CONFIGURATION SET 1: CONSERVATIVE (RECOMMENDED FOR FIRST RUN)
# ==============================================================================
# Best for: Small-medium datasets, noisy data, avoiding overfitting
# Expected: Slower training, better generalization, val_auc 0.65-0.70

conservative_config = {
    'model_parameters': {
        'wavenet_filters': 32,
        'wavenet_blocks': 4,
        'wavenet_layers_per_block': 3,
        'conv_filters': [64, 128, 256],
        'lstm_units': [256, 128],
        'attention_units': 128,
        'dropout_rate': 0.35,  # Slightly higher
        'l2_reg': 0.0001
    },
    'training_parameters': {
        'learning_rate': 0.0003,  # Lower for stability
        'beta_1': 0.9,
        'beta_2': 0.999,
        'clipnorm': 1.0,  # Gradient clipping
        'batch_size': 32,  # Smaller batches
        'max_epochs': 150,
        'steps_per_epoch': None,  # Use all data
        'early_stopping_patience': 30,  # More patient
        'reduce_lr_patience': 12,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
        'direction_loss_weight': 0.7,
        'volatility_loss_weight': 0.2,
        'magnitude_loss_weight': 0.3
    }
}

# ==============================================================================
# CONFIGURATION SET 2: BALANCED (BEST STARTING POINT)
# ==============================================================================
# Best for: Medium datasets, standard hardware, production use
# Expected: Good balance of speed and accuracy, val_auc 0.68-0.72

balanced_config = {
    'model_parameters': {
        'n_classes': 3,
        'wavenet_filters': 32,
        'wavenet_blocks': 3,
        'wavenet_layers_per_block': 3,
        'conv_filters': [64, 128, 256],
        'lstm_units': [256, 128],
        'attention_units': 128,
        'dropout_rate': 0.3,
        'l2_reg': 0.01
    },
    'training_parameters': {
        'learning_rate': 0.0001,  
        'beta_1': 0.9,
        'beta_2': 0.999,
        'clipnorm': 1.0,  # Gradient clipping
        'batch_size': 64,  # Standard
        'max_epochs': 150,
        'steps_per_epoch': None,  # Use all data
        'early_stopping_patience': 150,
        'reduce_lr_patience': 150,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
        'direction_loss_weight': 1.0,
        'volatility_loss_weight': 0.5,
        'magnitude_loss_weight': 0.5,
        'label_smoothing': 0.1  # Prevents overconfident predictions on noisy data
    }
}

# ==============================================================================
# CONFIGURATION SET 3: AGGRESSIVE (FAST ITERATION)
# ==============================================================================
# Best for: Large datasets (>200k), strong GPU, clean data
# Expected: Fast training, highest potential, val_auc 0.70-0.75

aggressive_config = {
    'model_parameters': {
        'wavenet_filters': 32,
        'wavenet_blocks': 4,
        'wavenet_layers_per_block': 3,
        'conv_filters': [64, 128, 256],
        'lstm_units': [256, 128],
        'attention_units': 128,
        'dropout_rate': 0.4,  # Lower dropout
        'l2_reg': 0.0005  # Less regularization
    },
    'training_parameters': {
        'learning_rate': 0.003,  # Standard Adam
        'beta_1': 0.9,
        'beta_2': 0.999,
        'clipnorm': 1.0,  # Gradient clipping
        'batch_size': 128,  # Larger batches
        'max_epochs': 200,
        'steps_per_epoch': None,  # Use all data
        'early_stopping_patience': 20,
        'reduce_lr_patience': 10,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
        'direction_loss_weight': 0.7,
        'volatility_loss_weight': 0.2,
        'magnitude_loss_weight': 0.3
    }
}

# ==============================================================================
# CONFIGURATION SET 4: WITH LR SCHEDULE (ADVANCED)
# ==============================================================================
# Uses cosine decay with warmup for optimal convergence

import tensorflow as tf
import numpy as np

def create_warmup_cosine_decay(initial_lr=0.0001, 
                               peak_lr=0.001,
                               warmup_epochs=10,
                               total_epochs=150,
                               steps_per_epoch=100):
    """
    Creates a learning rate schedule with warmup and cosine decay
    
    Phase 1: Linear warmup (0 -> warmup_epochs)
    Phase 2: Cosine decay (warmup_epochs -> total_epochs)
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(warmup_steps, tf.float32)
        total_steps_f = tf.cast(total_steps, tf.float32)
        
        if step < warmup_steps_f:
            # Linear warmup
            lr = initial_lr + (peak_lr - initial_lr) * (step / warmup_steps_f)
        else:
            # Cosine decay
            progress = (step - warmup_steps_f) / (total_steps_f - warmup_steps_f)
            lr = initial_lr + (peak_lr - initial_lr) * 0.5 * (
                1.0 + tf.cos(np.pi * progress)
            )
        return lr
    
    return tf.keras.optimizers.schedules.LearningRateSchedule(lr_schedule)

advanced_config = {
    'model_parameters': {
        'wavenet_filters': 32,
        'wavenet_blocks': 4,
        'wavenet_layers_per_block': 3,
        'conv_filters': [64, 128, 256],
        'lstm_units': [256, 128],
        'attention_units': 128,
        'dropout_rate': 0.3,
        'l2_reg': 0.0001
    },
    'training_parameters': {
        'use_lr_schedule': True,
        'initial_lr': 0.0001,
        'peak_lr': 0.001,
        'warmup_epochs': 10,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'batch_size': 64,
        'max_epochs': 150,
        'early_stopping_patience': 25,
        'direction_loss_weight': 0.7,
        'volatility_loss_weight': 0.2,
        'magnitude_loss_weight': 0.3
    }
}

# ==============================================================================
# DATASET SIZE RECOMMENDATIONS
# ==============================================================================

def recommend_config(n_samples):
    """
    Recommend configuration based on dataset size
    
    Args:
        n_samples: Number of training samples
        
    Returns:
        Recommended config name
    """
    if n_samples < 20000:
        return "conservative"
    elif n_samples < 100000:
        return "balanced"
    else:
        return "aggressive"

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Example: Switch configuration in fin_training.py
    
    # Option 1: Use balanced config (recommended)
    config = balanced_config
    
    # Option 2: Auto-select based on dataset size
    n_training_samples = 13000  # Replace with actual count
    config_name = recommend_config(n_training_samples)
    config = {
        'conservative': conservative_config,
        'balanced': balanced_config,
        'aggressive': aggressive_config
    }[config_name]
    
    print(f"Using {config_name} configuration")
    print(f"Learning rate: {config['training_parameters']['learning_rate']}")
    print(f"Batch size: {config['training_parameters']['batch_size']}")
    print(f"Max epochs: {config['training_parameters']['max_epochs']}")

# ==============================================================================
# MONITORING GUIDELINES
# ==============================================================================

"""
After training starts, monitor these metrics:

1. VALIDATION DIRECTION AUC (primary metric):
   - Target: > 0.65 (good for financial data)
   - Excellent: > 0.70
   - Outstanding: > 0.75

2. TRAIN vs VAL LOSS:
   - Healthy: val_loss within 20% of train_loss
   - Overfitting: val_loss > 1.5x train_loss
   - Underfitting: Both losses stay high

3. LEARNING RATE REDUCTIONS:
   - Expected: 2-4 reductions during training
   - Too many (>5): Initial LR too high
   - None: Initial LR too low or patience too long

4. EARLY STOPPING:
   - Ideal: Stops at 60-100 epochs
   - Too early (<30): Increase patience or LR
   - Never stops: Reduce max_epochs or check data

ADJUST BASED ON OBSERVATIONS:
- Val AUC plateaus early → Increase LR or model capacity
- Overfitting → Decrease LR, increase dropout, reduce capacity
- Underfitting → Increase LR or model capacity
- Training too slow → Increase batch size or LR
"""

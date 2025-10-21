import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from utils_train import F1Score
import utils 
import learning_data
import utils_plot
from wavenet_model import build_wavenet_model
import pickle
import glob
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
import random as rn
import csv
import io

data_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/data_modified_users_unfiltered'
# Define checkpoint directory
#checkpoint_dir = "/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/checkpoints_wave"
#checkpoint_path = os.path.join(checkpoint_dir, "ckpt")

# Path to where we want to save the training results
run_name = 'wavenet_8_bce_global_zero_include_fold'
base_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2'
save_path = os.path.join(base_path, f'run_{run_name}')

# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
#users = ['2','6','7','11']
users_test = users

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
                   'stroke_range':         6,       # Augments stroke labels in the dataset to include a range around detected peaks
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


# Parameters for training the CNN model
training_parameters = {'swim_style_lr':   5.0e-6,  # Constant for swim style
                       'stroke_lr':       5.0e-6,
                       'common_lr':       5.0e-6,
                       'beta_1':          0.9,
                       'beta_2':          0.999,
                       'batch_size':      64,
                       'max_epochs':      400,      # Keeping small for quick testing
                       'steps_per_epoch': 100,      # Keeping small for quick testing
                       'noise_std':       0.01,    # Noise standard deviation for data augmentation
                       'mirror_prob':     None,     # Probability of reversing a window for data augmentation
                       'random_rot_deg':  None,      # [-30, 30] is the range of rotation degrees we sample for each
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



def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
       # p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_loss = -alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            while len(sample_weight.shape) < len(focal_loss.shape):
                sample_weight = tf.expand_dims(sample_weight, -1)
            focal_loss = focal_loss * sample_weight
        
        focal_loss = tf.reduce_mean(focal_loss, axis=[1, 2])  # mean over timesteps and features
        return tf.reduce_mean(focal_loss)  # mean over batch

    return focal_loss_fn

# --- Learning Rate Scheduler ---
class BalancedAdaptiveLearningRateSchedule_old:
    def __init__(self, initial_common_lr=2e-5, initial_style_lr=1e-4, initial_stroke_lr=1e-4,
                 adjustment_factor=1.1, tolerance=0.05, min_lr=1e-6, max_lr=1e-2):
      #  self.initial_common_lr = initial_common_lr
        self.initial_style_lr = initial_style_lr
        self.initial_stroke_lr = initial_stroke_lr
        self.adjustment_factor = adjustment_factor
        self.tolerance = tolerance
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.last_style_loss = float('inf')
        self.last_stroke_loss = float('inf')
        
    def __call__(self, style_loss, stroke_loss):
        # Adjust learning rates based on loss changes
        style_change = (self.last_style_loss - style_loss) / (self.last_style_loss + 1e-8)
        stroke_change = (self.last_stroke_loss - stroke_loss) / (self.last_stroke_loss + 1e-8)
        
        # Update tracked losses
        self.last_style_loss = style_loss
        self.last_stroke_loss = stroke_loss
        
        # Default: keep current rates
       # common_lr = self.initial_common_lr
        style_lr = self.initial_style_lr
        stroke_lr = self.initial_stroke_lr
        
        # Adjust if needed
        if style_change < -self.tolerance:  # Style loss increased
            style_lr /= self.adjustment_factor
        elif style_change > self.tolerance:  # Style loss decreased significantly
            style_lr *= self.adjustment_factor
            
        if stroke_change < -self.tolerance:  # Stroke loss increased
            stroke_lr /= self.adjustment_factor
        elif stroke_change > self.tolerance:  # Stroke loss decreased significantly
            stroke_lr *= self.adjustment_factor
        
        # Ensure learning rates stay within bounds
        #common_lr = max(min(common_lr, self.max_lr), self.min_lr)
        style_lr = max(min(style_lr, self.max_lr), self.min_lr)
        stroke_lr = max(min(stroke_lr, self.max_lr), self.min_lr)
        
        return style_lr, stroke_lr

class BalancedAdaptiveLearningRateSchedule:
    def __init__(self, initial_common_lr=2e-5, initial_style_lr=1e-4, initial_stroke_lr=1e-4,
                 adjustment_factor=1.2, tolerance=0.05, min_lr=1e-6, 
                 max_common_lr=1e-3, max_style_lr=5e-3, max_stroke_lr=5e-4,
                 common_lr_sensitivity=0.8):
        # Initialize learning rates
        self.common_lr = initial_common_lr
        self.style_lr = initial_style_lr
        self.stroke_lr = initial_stroke_lr
        
        # Adjustment parameters
        self.adjustment_factor = adjustment_factor
        self.stroke_adjustment_factor = min(adjustment_factor, 1.1)  # More conservative for stroke
        self.tolerance = tolerance
        self.common_lr_sensitivity = common_lr_sensitivity
        
        # Learning rate boundaries
        self.min_lr = min_lr
        self.max_common_lr = max_common_lr
        self.max_style_lr = max_style_lr
        self.max_stroke_lr = max_stroke_lr  # Lower cap for stroke task
        
        # Loss tracking
        self.last_style_loss = float('inf')
        self.last_stroke_loss = float('inf')
        self.loss_history = []  # Track recent loss values for volatility detection
        self.update_count = 0  # Track number of updates for logging
        
    def __call__(self, style_loss, stroke_loss):
        self.update_count += 1
        
        # Emergency circuit breaker for large loss increases
        if self.last_stroke_loss != float('inf') and stroke_loss > 1.5 * self.last_stroke_loss:
            self.stroke_lr /= 2.0  # Halve the learning rate immediately
            print(f"⚠️ Emergency stroke LR reduction! Loss increased by {stroke_loss/self.last_stroke_loss:.2f}x. New LR: {self.stroke_lr:.6f}")
            
            # Update tracked losses but don't adjust other rates
            self.last_style_loss = style_loss
            self.last_stroke_loss = stroke_loss
            self.loss_history.append((style_loss, stroke_loss))
            if len(self.loss_history) > 5:
                self.loss_history.pop(0)
                
            return self.common_lr, self.style_lr, self.stroke_lr
        
        # Calculate loss changes
        if self.last_style_loss == float('inf') or self.last_stroke_loss == float('inf'):
            # First call - no changes to calculate
            style_change = 0.0
            stroke_change = 0.0
        else:
            style_change = (self.last_style_loss - style_loss) / (self.last_style_loss + 1e-8)
            stroke_change = (self.last_stroke_loss - stroke_loss) / (self.last_stroke_loss + 1e-8)

        # Store current losses for history tracking
        self.loss_history.append((style_loss, stroke_loss))
        if len(self.loss_history) > 5:
            self.loss_history.pop(0)
        
        # Update tracked losses
        self.last_style_loss = style_loss
        self.last_stroke_loss = stroke_loss
        
        # Adjust style learning rate
        if style_change < -self.tolerance:  # Style loss increased
            self.style_lr /= self.adjustment_factor
            print(f"📉 Style loss increased: reducing style LR to {self.style_lr:.6f}")
        elif style_change > self.tolerance:  # Style loss decreased significantly
            prev_lr = self.style_lr
            self.style_lr *= self.adjustment_factor
            print(f"📈 Style loss decreased: increasing style LR from {prev_lr:.6f} to {self.style_lr:.6f}")
            
        # Adjust stroke learning rate - more conservative increases
        if stroke_change < -self.tolerance:  # Stroke loss increased
            self.stroke_lr /= self.adjustment_factor
            print(f"📉 Stroke loss increased: reducing stroke LR to {self.stroke_lr:.6f}")
        elif stroke_change > self.tolerance:  # Stroke loss decreased significantly
            prev_lr = self.stroke_lr
            self.stroke_lr *= self.stroke_adjustment_factor  # More conservative increase
            print(f"📈 Stroke loss decreased: increasing stroke LR from {prev_lr:.6f} to {self.stroke_lr:.6f}")
        
        # Adjust common layers based on combined task performance
        combined_change = (style_change + stroke_change) / 2
        
        # 1. Adjust based on overall direction of losses
        if combined_change < -self.tolerance:  # Both tasks worsening on average
            self.common_lr /= self.adjustment_factor
            print(f"📉 Both losses worsening: reducing common LR to {self.common_lr:.6f}")
        elif combined_change > self.tolerance:  # Both tasks improving on average
            prev_lr = self.common_lr
            self.common_lr *= (self.adjustment_factor ** 0.5)  # More conservative increase
            print(f"📈 Both losses improving: increasing common LR from {prev_lr:.6f} to {self.common_lr:.6f}")
        
        # 2. Adjust for loss imbalance - if one task is doing much worse than the other
        loss_ratio = style_loss / (stroke_loss + 1e-8)
        if loss_ratio > 2.0 or loss_ratio < 0.5:  # One task is doing much worse
            self.common_lr *= 0.95  # Slight reduction to help stabilize
            print(f"⚖️ Loss imbalance detected (ratio={loss_ratio:.2f}): reducing common LR slightly to {self.common_lr:.6f}")
        
        # 3. Check for loss volatility
        if len(self.loss_history) >= 3:
            is_volatile = self._check_volatility()
            if is_volatile:
                self.common_lr *= 0.9  # Reduce LR when losses are oscillating
                print(f"🔄 Loss volatility detected: reducing common LR to {self.common_lr:.6f}")
        
        # Ensure learning rates stay within bounds (task-specific caps)
        self.common_lr = max(min(self.common_lr, self.max_common_lr), self.min_lr)
        self.style_lr = max(min(self.style_lr, self.max_style_lr), self.min_lr)
        self.stroke_lr = max(min(self.stroke_lr, self.max_stroke_lr), self.min_lr)
        
        # Print summary every 5 updates
        if self.update_count % 5 == 0:
            print(f"📊 Learning rates - Common: {self.common_lr:.6f}, Style: {self.style_lr:.6f}, Stroke: {self.stroke_lr:.6f}")
        
        return self.common_lr, self.style_lr, self.stroke_lr
    
    def _check_volatility(self):
        """Check if the losses are oscillating (sign of instability)"""
        if len(self.loss_history) < 3:
            return False
            
        # Check if losses are going up and down repeatedly
        style_directions = []
        stroke_directions = []
        
        for i in range(1, len(self.loss_history)):
            prev_style, prev_stroke = self.loss_history[i-1]
            curr_style, curr_stroke = self.loss_history[i]
            
            style_directions.append(1 if curr_style > prev_style else -1)
            stroke_directions.append(1 if curr_stroke > prev_stroke else -1)
        
        # Count direction changes
        style_changes = sum(1 for i in range(1, len(style_directions)) if style_directions[i] != style_directions[i-1])
        stroke_changes = sum(1 for i in range(1, len(stroke_directions)) if stroke_directions[i] != stroke_directions[i-1])
        
        # If we see frequent direction changes, the loss is volatile
        return (style_changes + stroke_changes) >= 2

class CombinedMetricEarlyStopping_old:
    """Custom early stopping that monitors multiple metrics with weights for multi-task learning."""
    
    def __init__(self, 
                 monitor_metrics={'val_style_accuracy_weighted': 0.5, 'val_stroke_f1_weighted': 0.5},
                 min_delta=0.001,
                 patience=10,
                 verbose=1,
                 mode='max',
                 restore_best_weights=True):
        self.monitor_metrics = monitor_metrics  # Dict of metrics and their weights
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Initialize tracking variables
        self.model = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.model_stop_training = False
    
    def set_model(self, model):
        """Store reference to model being trained."""
        self.model = model
        if self.restore_best_weights:
            self.best_weights = self.model.get_weights()
    
    def on_train_begin(self, logs=None):
        """Reset state at the start of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('-inf') if self.mode == 'max' else float('inf')
        self.model_stop_training = False
    
    def _compute_combined_metric(self, logs):
        if not logs:
            return None
        
        # Extract metrics and weights
        keys = list(self.monitor_metrics.keys())
        if len(keys) != 2:
            if self.verbose > 0:
                print("Warning: Harmonic mean implementation expects exactly 2 metrics.")
            return None
        
        metric1_name, metric2_name = keys
        alpha = self.monitor_metrics[metric1_name]
        beta = 1 - alpha
        
        metric1 = logs.get(metric1_name, 0.0)
        metric2 = logs.get(metric2_name, 0.0)
        
        # Avoid division by zero
        if metric1 == 0 or metric2 == 0:
            return 0.0
        
        # Weighted harmonic mean
        combined_score = 2 / ((alpha / metric1) + (beta / metric2))
        return combined_score

    def compute_combined_metric(self, logs):
        """Public method to compute combined metric from logs dict."""
        return self._compute_combined_metric(logs)    

    def on_epoch_end(self, epoch, logs=None):
        """Check at epoch end if training should stop."""
        combined_score = self._compute_combined_metric(logs)
        
        if combined_score is None:
            return
        
        if self.verbose > 0:
            metrics_str = ", ".join([f"{k}={logs.get(k, 'N/A'):.4f}" for k in self.monitor_metrics.keys() if k in logs])
            print(f"Epoch {epoch+1}: {metrics_str}, combined={combined_score:.4f}")
        
        # Check if score improved
        improved = False
        if self.mode == 'max' and combined_score > self.best_score + self.min_delta:
            improved = True
        elif self.mode == 'min' and combined_score < self.best_score - self.min_delta:
            improved = True
            
        if improved:
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: Combined metric improved from {self.best_score:.4f} to {combined_score:.4f}")
            self.best_score = combined_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: Combined metric {combined_score:.4f} did not improve from {self.best_score:.4f}. {self.patience - self.wait} epochs remaining.")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model_stop_training = True
                if self.verbose > 0:
                    print(f"Epoch {epoch+1}: early stopping triggered")
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the best epoch")
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        """Final report at end of training."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping occurred at epoch {self.stopped_epoch+1}")
        return self.model_stop_training

class CombinedMetricEarlyStopping:
    """Custom early stopping that monitors multiple metrics with dynamic weighted harmonic mean for multi-task learning."""
    
    def __init__(self, 
                 monitor_metrics={'val_style_accuracy_weighted': 0.5, 'val_stroke_f1_weighted': 0.5},
                 min_delta=0.001,
                 patience=10,
                 verbose=1,
                 mode='max',
                 restore_best_weights=True,
                 adjust_weights=True,
                 adjustment_window=5,
                 weight_increment=0.05,
                 weight_decrement=0.05):
        self.monitor_metrics = monitor_metrics  # Dict of metrics and their weights
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.adjust_weights = adjust_weights
        self.adjustment_window = adjustment_window
        self.weight_increment = weight_increment
        self.weight_decrement = weight_decrement
        
        # Initialize tracking variables
        self.model = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.model_stop_training = False
        
        # Store history of metric values
        self.metric_history = {key: [] for key in self.monitor_metrics.keys()}
    
    def set_model(self, model):
        """Store reference to model being trained."""
        self.model = model
        if self.restore_best_weights:
            self.best_weights = self.model.get_weights()
    
    def on_train_begin(self, logs=None):
        """Reset state at the start of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('-inf') if self.mode == 'max' else float('inf')
        self.model_stop_training = False
        self.metric_history = {key: [] for key in self.monitor_metrics.keys()}
    
    def update_weights(self):
        """Adjust weights dynamically based on recent metric improvements."""
        improvements = {}
        for metric, history in self.metric_history.items():
            if len(history) < self.adjustment_window + 1:
                improvements[metric] = None
                continue
            improvements[metric] = history[-1] - history[-(self.adjustment_window + 1)]
        
        keys = list(self.monitor_metrics.keys())
        m1, m2 = keys[0], keys[1]
        
        if improvements[m1] is not None and improvements[m2] is not None:
            if improvements[m1] > self.min_delta and improvements[m2] <= self.min_delta:
                self.monitor_metrics[m1] = min(1.0, self.monitor_metrics[m1] + self.weight_increment)
                self.monitor_metrics[m2] = max(0.0, self.monitor_metrics[m2] - self.weight_decrement)
            elif improvements[m2] > self.min_delta and improvements[m1] <= self.min_delta:
                self.monitor_metrics[m2] = min(1.0, self.monitor_metrics[m2] + self.weight_increment)
                self.monitor_metrics[m1] = max(0.0, self.monitor_metrics[m1] - self.weight_decrement)
            # Normalize to sum to 1
            total = self.monitor_metrics[m1] + self.monitor_metrics[m2]
            self.monitor_metrics[m1] /= total
            self.monitor_metrics[m2] /= total
            
            if self.verbose > 0:
                print(f"✅ Adjusted monitor metric weights: {self.monitor_metrics}")
    
    def _compute_combined_metric(self, logs):
        """Compute weighted harmonic mean of monitored metrics."""
        if not logs:
            return None
        
        keys = list(self.monitor_metrics.keys())
        if len(keys) != 2:
            if self.verbose > 0:
                print("Warning: Harmonic mean implementation expects exactly 2 metrics.")
            return None
        
        metric1_name, metric2_name = keys
        alpha = self.monitor_metrics[metric1_name]
        beta = 1 - alpha
        
        metric1 = logs.get(metric1_name, 0.0)
        metric2 = logs.get(metric2_name, 0.0)
        
        if metric1 == 0 or metric2 == 0:
            return 0.0
        
        combined_score = 2 / ((alpha / metric1) + (beta / metric2))
        return combined_score
    
    def compute_combined_metric(self, logs):
        """Public method to compute combined metric from logs dict."""
        return self._compute_combined_metric(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        """Epoch-end hook to update weights and decide on early stopping."""
        logs = logs or {}
        
        # Update metric history
        for metric in self.monitor_metrics.keys():
            val = logs.get(metric)
            if val is not None:
                self.metric_history[metric].append(val)
        
        # Adjust weights dynamically every adjustment_window epochs
        if self.adjust_weights and epoch > 0 and (epoch + 1) % self.adjustment_window == 0:
            self.update_weights()
        
        combined_score = self._compute_combined_metric(logs)
        if combined_score is None:
            return
        
        if self.verbose > 0:
            metrics_str = ", ".join([f"{k}={logs.get(k, 'N/A'):.4f}" for k in self.monitor_metrics.keys() if k in logs])
            print(f"Epoch {epoch+1}: {metrics_str}, combined={combined_score:.4f}")
        
        improved = False
        if self.mode == 'max' and combined_score > self.best_score + self.min_delta:
            improved = True
        elif self.mode == 'min' and combined_score < self.best_score - self.min_delta:
            improved = True
        
        if improved:
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: Combined metric improved from {self.best_score:.4f} to {combined_score:.4f}")
            self.best_score = combined_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: Combined metric {combined_score:.4f} did not improve from {self.best_score:.4f}. {self.patience - self.wait} epochs remaining.")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model_stop_training = True
                if self.verbose > 0:
                    print(f"Epoch {epoch+1}: early stopping triggered")
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the best epoch")
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        """Final report at end of training."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping occurred at epoch {self.stopped_epoch+1}")
        return self.model_stop_training

class CustomCSVLogger:
    """CSV Logger for multi-task learning with support for early stopping and learning rate scheduling."""
    
    def __init__(self, filename, separator=',', append=False):
        self.filename = filename
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.file = None
        self.model = None
        
        # Create directory if it doesn't exist
        dirname = os.path.dirname(self.filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            
        # Prepare file and writer
        file_exists = os.path.isfile(self.filename)
        if self.append and file_exists:
            with open(self.filename, 'r') as f:
                reader = csv.reader(f, delimiter=self.sep)
                self.keys = next(reader)
    
    def set_model(self, model):
        self.model = model
    
    def on_train_begin(self, logs=None):
        if self.append and os.path.isfile(self.filename):
            self.file = open(self.filename, 'a')
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys, delimiter=self.sep)
        else:
            self.file = open(self.filename, 'w')
            self.writer = None
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Add epoch and timestamp
        logs['epoch'] = epoch + 1
        logs['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format learning rates for better readability
        if 'lr_common' in logs and logs['lr_common'] is not None:
            logs['lr_common'] = f"{logs['lr_common']:.8f}"
        if 'lr_style' in logs and logs['lr_style'] is not None:
            logs['lr_style'] = f"{logs['lr_style']:.8f}"
        if 'lr_stroke' in logs and logs['lr_stroke'] is not None:
            logs['lr_stroke'] = f"{logs['lr_stroke']:.8f}"
            
        # Initialize writer if first epoch
        if self.writer is None:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys, delimiter=self.sep)
            self.writer.writeheader()
        
        # Handle metrics not present in current logs but in header
        row_dict = {k: logs.get(k, 'NA') for k in self.keys}
        
        # Write the row
        self.writer.writerow(row_dict)
        self.file.flush()
        
    def on_train_end(self, logs=None):
        if self.file is not None:
            self.file.close()
            self.writer = None
            print(f"Training history saved to {self.filename}")


# --- Main Training Function ---
def train_multitask_model(model, train_gen, val_data=None, training_parameters=None, 
                           log_dir=None, experiment_save_path=None, checkpoint_path=None):
    """
    Training function for multi-task model using custom training loop
    """
    epochs = training_parameters['max_epochs']
    steps_per_epoch = training_parameters['steps_per_epoch']
    # Create optimizers
    #common_optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['common_lr'])
    style_optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['swim_style_lr'])
    stroke_optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['stroke_lr'])
    
    # Create loss functions
    style_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    #stroke_loss_fn = focal_loss(gamma=2.0, alpha=0.5)
    """
    stroke_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(    
                       # apply_class_balancing=True,
                        alpha=0.6,
                        gamma=2.0,
                        from_logits=False,
                        label_smoothing=0.05
    )
    """
    stroke_loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    # Create metrics
    style_metric = tf.keras.metrics.CategoricalAccuracy(name='style_accuracy')
    stroke_metric = F1Score(threshold=0.5, name='stroke_f1')

    style_metric_weighted = tf.keras.metrics.CategoricalAccuracy(name='style_accuracy_weighted')
    stroke_metric_weighted = F1Score(threshold=0.5, name='stroke_f1_weighted')
    
    # Create checkpoint object
    if checkpoint_path:
        epoch_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="epoch_counter")
        step_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="step_counter")
        log_dir_var = tf.Variable("", dtype=tf.string, trainable=False, name="log_dir")
        
        checkpoint = tf.train.Checkpoint(
            model=model,
            common_optimizer=common_optimizer,
            style_optimizer=style_optimizer,
            stroke_optimizer=stroke_optimizer,
            epoch_counter=epoch_counter,
            step_counter=step_counter,
            log_dir_var=log_dir_var
        )
        
        # Restore checkpoint if it exists
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
        if latest_ckpt:
            checkpoint.restore(latest_ckpt)
            print(f"✅ Checkpoint restored from {latest_ckpt}")
            print(f"ℹ️ Resuming training from epoch {epoch_counter.numpy()} and step {step_counter.numpy()}")
            start_epoch = epoch_counter.numpy()
            start_step = step_counter.numpy()
            
            # Restore log_dir if saved
            if log_dir_var.numpy():
                log_dir = log_dir_var.numpy().decode('utf-8')
                print(f"ℹ️ Resuming TensorBoard logging to {log_dir}")
        else:
            start_epoch = 0
            start_step = 0
            if log_dir:
                log_dir_var.assign(log_dir)
    else:
        start_epoch = 0
        start_step = 0
    
    # Create TensorBoard writer

    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Learning rate scheduler
    lr_scheduler = BalancedAdaptiveLearningRateSchedule(
        initial_common_lr=training_parameters['common_lr'],
        initial_style_lr=training_parameters['swim_style_lr'],
        initial_stroke_lr=training_parameters['stroke_lr'],
        adjustment_factor=1.1
    )
    
    # Initialize metrics history
    history = {
        'style_loss': [], 'stroke_loss': [],
        'style_accuracy': [], 'stroke_f1': [],
        'style_accuracy_weighted': [], 'stroke_f1_weighted': [],
        'val_style_loss': [], 'val_stroke_loss': [],
        'val_style_accuracy': [], 'val_style_accuracy_weighted': [], 
        'val_stroke_f1': [], 'val_stroke_f1_weighted': [],
        'val_stroke_similarity': [], 'val_confusion_data': []     
    }

    
    # Training step function
    @tf.function
    def train_step(x, y_style, y_stroke, style_weights=None, stroke_weights=None):
        lambda_style = 1.0
        lambda_stroke = 1.0

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            y_pred = model(x, training=True)
            swim_output, stroke_output = y_pred
            
            # Calculate losses with sample weights
            style_loss = style_loss_fn(
                y_style, 
                swim_output,
                sample_weight=None
            )
            """
            # Ensure correct shape for stroke weights
            if stroke_weights is not None:
                # If shape is (batch, seq_len, 1), squeeze to (batch, seq_len)
                if tf.rank(stroke_weights) > 2:
                    stroke_weights = tf.squeeze(stroke_weights, axis=-1)
            """
            stroke_loss = stroke_loss_fn(
                y_stroke, 
                stroke_output,
                sample_weight=None
            )
            
            # Get variables for each part of the model
            common_vars = [v for v in model.trainable_variables 
                           if 'swim_style' not in v.name and 'stroke_label' not in v.name]
            style_vars = [v for v in model.trainable_variables if 'swim_style' in v.name]
            stroke_vars = [v for v in model.trainable_variables if 'stroke_label' in v.name]

        style_grads = tape.gradient(style_loss, common_vars + style_vars)
        stroke_grads = tape.gradient(stroke_loss, common_vars + stroke_vars)
        # Clean up the persistent tape
        del tape
        # Clip gradients
        style_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in style_grads]
        stroke_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in stroke_grads]

        style_optimizer.apply_gradients(zip(style_grads, common_vars + style_vars))
        stroke_optimizer.apply_gradients(zip(stroke_grads, common_vars + stroke_vars))

        
        # Update metrics
        style_metric.update_state(y_style, swim_output)
        stroke_metric.update_state(y_stroke, stroke_output)
        style_metric_weighted.update_state(y_style, swim_output, sample_weight=style_weights)
        stroke_metric_weighted.update_state(y_stroke, stroke_output, sample_weight=stroke_weights)
        
        
        return {
            'style_loss': style_loss,
            'stroke_loss': stroke_loss,
            'style_accuracy': style_metric.result(),
            'stroke_f1': stroke_metric.result(),
            'style_accuracy_weighted': style_metric_weighted.result(),
            'stroke_f1_weighted': stroke_metric_weighted.result()
        }
    
    def validate_model(x_val, y_style_val, y_stroke_val, style_weights_val=None, stroke_weights_val=None):
        # Forward pass
        y_pred = model(x_val, training=False)
        swim_output, stroke_output = y_pred
        """
        # Ensure correct shape for stroke weights
        if stroke_weights_val is not None:
            # If shape is (batch, seq_len, 1), squeeze to (batch, seq_len)
            if len(tf.shape(stroke_weights_val)) > 2:
                stroke_weights_val_for_loss = tf.squeeze(stroke_weights_val, axis=-1)
            else:
                stroke_weights_val_for_loss = stroke_weights_val
            stroke_weights_val_for_loss = tf.cast(stroke_weights_val_for_loss, dtype=tf.float32)
        else:
            stroke_weights_val_for_loss = None
        """
        # Calculate losses with properly shaped weights
        style_loss = style_loss_fn(y_style_val, swim_output, sample_weight=None)
        stroke_loss = stroke_loss_fn(y_stroke_val, stroke_output, sample_weight=None)


        # Weighted style accuracy
        style_accuracy = tf.keras.metrics.categorical_accuracy(y_style_val, swim_output)
        if style_weights_val is not None:
            style_weights_val = tf.cast(style_weights_val, tf.float32)
            style_accuracy_weighted = tf.reduce_sum(style_accuracy * style_weights_val) / tf.reduce_sum(style_weights_val)
        else:
            style_accuracy_weighted = tf.reduce_mean(style_accuracy)

        # Unweighted style accuracy (no sample weights)
        style_accuracy = tf.reduce_mean(style_accuracy)

        # Prepare stroke predictions for sklearn metrics
        y_pred_stroke = (stroke_output.numpy() > 0.5).astype(np.float32)

        # Weighted stroke F1
        if stroke_weights_val is not None:
            y_stroke_flat = y_stroke_val.reshape(-1)
            y_pred_stroke_flat = y_pred_stroke.reshape(-1)
            weights_flat = stroke_weights_val.reshape(-1)
            real_stroke_count = np.sum(y_stroke_flat, dtype=np.int32)
            fake_strokes = np.where(y_pred_stroke_flat > 0.5, 1, 0)

            fake_stroke_count = np.sum(fake_strokes)
            similarity = 100 - (np.abs(real_stroke_count - fake_stroke_count) / (real_stroke_count + 1e-8)) * 100
            print(f"\nReal Stroke Count: {real_stroke_count} | Fake Stroke Count: {fake_stroke_count} | Similarity: {similarity:.2f}%")

            mask = weights_flat > 0
            y_stroke_filtered = y_stroke_flat[mask]
            y_pred_stroke_filtered = y_pred_stroke_flat[mask]

            if len(y_stroke_filtered) > 0:
                stroke_f1_weighted = f1_score(y_stroke_filtered, y_pred_stroke_filtered, average='weighted')
            else:
                stroke_f1_weighted = 0.0
        else:
            #stroke_f1_weighted = f1_score(y_stroke_val.reshape(-1), y_pred_stroke.reshape(-1), average='weighted')
            # Unweighted stroke F1 (no sample weights)
            stroke_f1 = f1_score(y_stroke_val.reshape(-1), y_pred_stroke.reshape(-1), average='macro')
        # Add per-class accuracy:
        style_preds = np.argmax(swim_output.numpy(), axis=1)
        style_true = np.argmax(y_style_val, axis=1)

        # Calculate confusion matrix data
        confusion_data = {
            'predictions': style_preds.tolist(),  # Convert numpy to list
            'true_labels': style_true.tolist(),   # Convert numpy to list
        }

        for i in range(5):  # Assuming 5 style classes
            class_indices = np.where(style_true == i)[0]
            if len(class_indices) > 0:
                class_preds = style_preds[class_indices]
                class_accuracy = np.mean(class_preds == i)
                confusion_data[f'class_{i}_accuracy'] = float(class_accuracy)
                confusion_data[f'class_{i}_count'] = int(len(class_indices))


        return {
            'val_style_loss': style_loss.numpy(),
            'val_stroke_loss': stroke_loss.numpy(),
            'val_style_accuracy': style_accuracy.numpy(),
            'val_style_accuracy_weighted': style_accuracy_weighted.numpy(),
            'val_stroke_f1': stroke_f1,
            'val_stroke_f1_weighted': stroke_f1_weighted,
            'val_stroke_similarity': similarity,
            'val_confusion_data': confusion_data
        }

    # Create CSV Logger

    csv_file = os.path.join(experiment_save_path, f'training_history.csv')
    csv_logger = CustomCSVLogger(csv_file, append=False)
    csv_logger.set_model(model)
    csv_logger.on_train_begin()
    
    # Early stopping initialization (if using)
    early_stopping = CombinedMetricEarlyStopping(
        monitor_metrics={'val_style_accuracy_weighted': 0.2, 'val_stroke_f1_weighted': 0.8},
        min_delta=1e-5,  # smaller to detect subtle improvements
        patience=30,
        verbose=1,
        mode='max',
        restore_best_weights=True,
        adjust_weights=True,
        adjustment_window=2,
        weight_increment=0.1,
        weight_decrement=0.05
    )
    early_stopping.set_model(model)
    early_stopping.on_train_begin()
       
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Reset metrics
        style_metric.reset_state()
        stroke_metric.reset_state()
        style_metric_weighted.reset_state()
        stroke_metric_weighted.reset_state() 
        # Training
        epoch_metrics = {
            'style_loss': [], 
            'stroke_loss': [],
            'style_accuracy': [],
            'style_accuracy_weighted': [],
            'stroke_f1': [],
            'stroke_f1_weighted': []
        }
        
        for step in range(steps_per_epoch):
            if checkpoint_path:
                step_counter.assign_add(1)
                
            # Get batch of data
            x_batch, y_dict, sample_weights = next(train_gen)
            
            # Skip incomplete batches
          #  if x_batch.shape[0] < train_gen.batch_size:
           #     continue
            
            # Get labels and compute sample weights
            y_style = y_dict['swim_style_output']
            y_stroke = y_dict['stroke_label_output']
            style_weights = sample_weights['swim_style_output']
            stroke_weights = sample_weights['stroke_label_output']
            style_weights = tf.convert_to_tensor(style_weights, dtype=tf.float32)
            stroke_weights = tf.convert_to_tensor(stroke_weights, dtype=tf.float32)

            # Training step
            metrics = train_step(x_batch, y_style, y_stroke, style_weights, stroke_weights)
            
            # Record metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value.numpy() if hasattr(value, 'numpy') else value)
            
            # Print progress
            if step % 10 == 0:
                print(f"  Step {step}/{steps_per_epoch}: "
                      f"Style Loss: {metrics['style_loss']:.4f}, "
                      f"Stroke Loss: {metrics['stroke_loss']:.4f}, "
                      f"Style Acc: {metrics['style_accuracy']:.4f}, "
                      f"Style Acc Weighted: {metrics['style_accuracy_weighted']:.4f}, "
                      f"Stroke F1: {metrics['stroke_f1']:.4f}, "
                      f"Stroke F1 Weighted: {metrics['stroke_f1_weighted']:.4f}")
        
        # Compute epoch metrics
        for key in epoch_metrics:
            history[key].append(np.mean(epoch_metrics[key]))

        epoch_logs = {
            'style_loss': np.mean(epoch_metrics['style_loss']),
            'stroke_loss': np.mean(epoch_metrics['stroke_loss']),
            'style_accuracy': np.mean(epoch_metrics['style_accuracy']),
            'style_accuracy_weighted': np.mean(epoch_metrics['style_accuracy_weighted']),
            'stroke_f1': np.mean(epoch_metrics['stroke_f1']),
            'stroke_f1_weighted': np.mean(epoch_metrics['stroke_f1_weighted']),
            # Add learning rates
            'lr_style': style_optimizer.learning_rate.numpy(),
            'lr_stroke': stroke_optimizer.learning_rate.numpy(),
            'lr_common': common_optimizer.learning_rate.numpy() if common_optimizer else None
        }      
        # Validate if validation data is provided
        if val_data is not None:
            x_val, y_val_dict, y_val_weights = val_data
            
            # Get validation labels
            y_style_val = y_val_dict['swim_style_output']
            y_stroke_val = y_val_dict['stroke_label_output']
            
            # Get validation weights if available
            style_weights_val = y_val_weights['swim_style_output']#y_val_dict.get('style_weights', None)
            stroke_weights_val = y_val_weights['stroke_label_output']#y_val_dict.get('stroke_weights', None)
            
            # Run validation
            val_metrics = validate_model(
                x_val, y_style_val, y_stroke_val, 
                style_weights_val, stroke_weights_val
            )
            
            # Record validation metrics
            for key, value in val_metrics.items():
                history[key].append(value)
            
            print(f"  Validation: "
                  f"Style Loss: {val_metrics['val_style_loss']:.4f}, "
                  f"Stroke Loss: {val_metrics['val_stroke_loss']:.4f}, "
                  f"Style Acc: {val_metrics['val_style_accuracy']:.4f}, "
                  f"Style Acc Weighted: {val_metrics['val_style_accuracy_weighted']:.4f}, "
                  f"Stroke F1: {val_metrics['val_stroke_f1']:.4f}, "
                  f"Stroke F1 Weighted: {val_metrics['val_stroke_f1_weighted']:.4f}")
            # Clean print of class metrics only
            def print_class_metrics(confusion_data):
                # Define class names for better readability
                class_names = {
                    0: 'NULL',
                    1: 'Freestyle', 
                    2: 'Breaststroke', 
                    3: 'Backstroke', 
                    4: 'Butterfly'
                }
                
                print("Per-class Style Performance:")
                print("----------------------------")
                for i in range(5):  # For each class
                    if f'class_{i}_accuracy' in confusion_data and f'class_{i}_count' in confusion_data:
                        acc = confusion_data[f'class_{i}_accuracy']
                        count = confusion_data[f'class_{i}_count']
                        name = class_names[i]
                        print(f"  {name}: {acc*100:.1f}% accuracy ({count} samples)")
                print("----------------------------")

            # Use it like this
            print_class_metrics(val_metrics['val_confusion_data'])

            epoch_logs.update({
                'val_style_loss': val_metrics['val_style_loss'],
                'val_stroke_loss': val_metrics['val_stroke_loss'],
                'val_style_accuracy': val_metrics['val_style_accuracy'],
                'val_style_accuracy_weighted': val_metrics['val_style_accuracy_weighted'],
                'val_stroke_f1': val_metrics['val_stroke_f1'],
                'val_stroke_f1_weighted': val_metrics['val_stroke_f1_weighted'],
                'val_stroke_similarity': val_metrics['val_stroke_similarity'],
                'val_confusion_data': val_metrics['val_confusion_data']
            })
            # Compute combined metric score
            combined_score = early_stopping.compute_combined_metric(epoch_logs)
            if combined_score is not None:
                epoch_logs['combined_metric'] = combined_score
            
            # Check early stopping
            early_stopping.on_epoch_end(epoch, epoch_logs)
            if early_stopping.model_stop_training:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                # Log the early stopping event
                epoch_logs['early_stopping'] = True
                csv_logger.on_epoch_end(epoch, epoch_logs)
                break
        
        # Log metrics to CSV
        csv_logger.on_epoch_end(epoch, epoch_logs)

        # Update epoch counter
        if checkpoint_path:
            epoch_counter.assign(epoch + 1)
        
        if epoch > 0:  # Skip first epoch
            # Adjust learning rates if needed (every N epochs)
            new_common_lr, new_style_lr, new_stroke_lr = lr_scheduler(
                history['val_style_loss'][-1], 
                history['val_stroke_loss'][-1]
            )
            
            # Apply new learning rates
            common_optimizer.learning_rate.assign(new_common_lr)
            style_optimizer.learning_rate.assign(new_style_lr)
            stroke_optimizer.learning_rate.assign(new_stroke_lr)
        
            print(f"  Learning rates updated - Common: {new_common_lr:.2e}, "
                    f"Style: {new_style_lr:.2e}, Stroke: {new_stroke_lr:.2e}")
        
        # Save checkpoint
        if checkpoint_path and epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_path)
            print(f"✅ Checkpoint saved at epoch {epoch}")


        # TensorBoard logging
        with summary_writer.as_default():
            # Training metrics
            tf.summary.scalar("train/style_loss", history['style_loss'][-1], step=epoch)
            tf.summary.scalar("train/stroke_loss", history['stroke_loss'][-1], step=epoch)
            tf.summary.scalar("train/style_accuracy", history['style_accuracy'][-1], step=epoch)
            tf.summary.scalar("train/stroke_f1", history['stroke_f1'][-1], step=epoch)
            tf.summary.scalar("train/style_accuracy_weighted", history['style_accuracy_weighted'][-1], step=epoch)
            tf.summary.scalar("train/stroke_f1_weighted", history['stroke_f1_weighted'][-1], step=epoch)
            
            # Validation metrics if available
            if val_data is not None:
                tf.summary.scalar("val/style_loss", history['val_style_loss'][-1], step=epoch)
                tf.summary.scalar("val/stroke_loss", history['val_stroke_loss'][-1], step=epoch)
                tf.summary.scalar("val/style_accuracy", history['val_style_accuracy'][-1], step=epoch)
                tf.summary.scalar("val/style_accuracy_weighted", history['val_style_accuracy_weighted'][-1], step=epoch)

                tf.summary.scalar("val/stroke_f1", history['val_stroke_f1'][-1], step=epoch)
                tf.summary.scalar("val/stroke_f1_weighted", history['val_stroke_f1_weighted'][-1], step=epoch)

                tf.summary.scalar("val/combined_metric", combined_score, step=epoch)
                tf.summary.scalar("val/stroke_similarity", history['val_stroke_similarity'][-1], step=epoch)
                # Add confusion data visualization
                if 'val_confusion_data' in history and len(history['val_confusion_data']) > 0:
                    # Log each metric as a scalar
                    confusion_data = history['val_confusion_data'][-1]
                    for class_name, value in confusion_data.items():
                        if isinstance(value, (float, int)) and not isinstance(value, bool):
                            tf.summary.scalar(f"style_classes/{class_name}", value, step=epoch)  
                    
                    # Create and log full confusion matrix visualization
                    class_names = ['NULL', 'Freestyle', 'Breaststroke', 'Backstroke', 'Butterfly']
                    figure = plot_confusion_matrix_to_tensorboard(confusion_data, class_names)
                    tf.summary.image("style_metrics", plot_to_image(figure), step=epoch)

            # Learning rates
            tf.summary.scalar("lr/common", common_optimizer.learning_rate.numpy(), step=epoch)
            tf.summary.scalar("lr/style", style_optimizer.learning_rate.numpy(), step=epoch)
            tf.summary.scalar("lr/stroke", stroke_optimizer.learning_rate.numpy(), step=epoch)
            
            summary_writer.flush()

    # Clean up callbacks
    csv_logger.on_train_end()
    stopped = early_stopping.on_train_end()

    if checkpoint_path and os.path.exists(os.path.dirname(checkpoint_path)):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if epoch_counter.numpy() >= training_parameters['max_epochs'] or stopped:
            print(f"✅ Training completed successfully for current user. Cleaning up checkpoint files at {checkpoint_dir}")

            # Find all checkpoint files matching the prefix (e.g., "ckpt*")
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, 'ckpt*'))

            # Delete each checkpoint file
            for f in ckpt_files:
                try:
                    os.remove(f)
                    print(f"Deleted checkpoint file: {f}")
                except Exception as e:
                    print(f"Error deleting {f}: {e}")

            # Optional: If TensorFlow creates a 'checkpoint' metadata file, delete it too
            checkpoint_metadata = os.path.join(checkpoint_dir, 'checkpoint')
            if os.path.exists(checkpoint_metadata):
                try:
                    os.remove(checkpoint_metadata)
                    print(f"Deleted checkpoint metadata file: {checkpoint_metadata}")
                except Exception as e:
                    print(f"Error deleting {checkpoint_metadata}: {e}")

        else:
            print(f"⚠️ Training did not complete all epochs for current user. Keeping checkpoints at {checkpoint_dir}")
    return history

# --- Utility function to plot training history ---
def plot_training_history(history):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Changed to 3 rows
    
    # Plot losses
    axs[0, 0].plot(history['style_loss'], label='Train')
    if 'val_style_loss' in history and history['val_style_loss']:
        axs[0, 0].plot(history['val_style_loss'], label='Validation')
    axs[0, 0].set_title('Style Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    axs[0, 1].plot(history['stroke_loss'], label='Train')
    if 'val_stroke_loss' in history and history['val_stroke_loss']:
        axs[0, 1].plot(history['val_stroke_loss'], label='Validation')
    axs[0, 1].set_title('Stroke Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    
    # Plot metrics
    axs[1, 0].plot(history['style_accuracy'], label='Train')
    if 'val_style_accuracy' in history and history['val_style_accuracy']:
        axs[1, 0].plot(history['val_style_accuracy'], label='Validation')
    axs[1, 0].set_title('Style Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    
    axs[1, 1].plot(history['stroke_f1'], label='Train')
    if 'val_stroke_f1' in history and history['val_stroke_f1']:
        axs[1, 1].plot(history['val_stroke_f1'], label='Validation')
    axs[1, 1].set_title('Stroke F1 Score')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].legend()

    # Plot metrics
    axs[2, 0].plot(history['style_accuracy'], label='Train')
    if 'val_style_accuracy_weighted' in history and history['val_style_accuracy_weighted']:
        axs[2, 0].plot(history['val_style_accuracy_weighted'], label='Validation')
    axs[2, 0].set_title('Style Accuracy Weighted')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].legend()
    
    axs[2, 1].plot(history['stroke_f1'], label='Train')
    if 'val_stroke_f1_weighted' in history and history['val_stroke_f1_weighted']:
        axs[2, 1].plot(history['val_stroke_f1_weighted'], label='Validation')
    axs[2, 1].set_title('Stroke F1 Score Weighted')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('F1 Score')
    axs[2, 1].legend()  
    plt.tight_layout()
    return fig

def plot_confusion_matrix_to_tensorboard(confusion_data, class_names=None):
    """
    Plot comprehensive confusion matrix visualization from confusion_data
    
    Args:
        confusion_data: Dictionary with class metrics and raw predictions
        class_names: Optional list of class names
    
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    if class_names is None:
        class_names = ['NULL', 'Freestyle', 'Breaststroke', 'Backstroke', 'Butterfly']
    
    # Extract raw predictions and labels if available
    has_raw_data = 'predictions' in confusion_data and 'true_labels' in confusion_data
    
    if has_raw_data:
        try:
            y_pred = confusion_data['predictions']
            y_true = confusion_data['true_labels']
            
            # Safety check - ensure we have same number of predictions and true labels
            if len(y_pred) != len(y_true):
                print(f"Warning: Mismatch in predictions ({len(y_pred)}) and true labels ({len(y_true)})")
                has_raw_data = False
            else:
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
                
                # Calculate per-class metrics
                class_precision = []
                class_recall = []
                
                for i in range(len(class_names)):
                    # Precision = TP / (TP + FP)
                    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                    class_precision.append(precision)
                    
                    # Recall = TP / (TP + FN)
                    recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                    class_recall.append(recall)
        except Exception as e:
            print(f"Error processing raw prediction data: {e}")
            has_raw_data = False
    
    # Extract accuracy and count data for each class
    num_classes = len(class_names)
    class_accuracies = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for i in range(num_classes):
        acc_key = f'class_{i}_accuracy'
        count_key = f'class_{i}_count'
        
        if acc_key in confusion_data:
            class_accuracies[i] = confusion_data[acc_key]
        
        if count_key in confusion_data:
            class_counts[i] = confusion_data[count_key]
    
    # Create appropriately sized figure
    if has_raw_data:
        fig = plt.figure(figsize=(20, 8))  # Wider figure for 3 panels
        gs = plt.GridSpec(2, 3, height_ratios=[3, 1])
        
        # Main plots
        ax1 = plt.subplot(gs[0, 0])  # Accuracy bars
        ax2 = plt.subplot(gs[0, 1])  # Class distribution pie
        ax3 = plt.subplot(gs[0, 2])  # Confusion matrix
        
        # Precision/Recall subplot
        ax4 = plt.subplot(gs[1, :])  # Spans all columns in bottom row
    else:
        fig = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
    
    # Plot class accuracies
    bars = ax1.bar(range(len(class_names)), class_accuracies, color='skyblue')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Style Classification Accuracy by Class')
    ax1.set_ylabel('Accuracy')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bar, accuracy in zip(bars, class_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{accuracy:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot class distribution (sample counts)
    wedges, texts, autotexts = ax2.pie(
        class_counts, 
        labels=class_names, 
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=90,
        colors=plt.cm.tab10.colors[:len(class_names)]
    )
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title('Class Distribution in Validation Set')
    
    # Plot confusion matrix if we have raw data
    if has_raw_data:
        cax = ax3.matshow(cm, cmap='Blues')
        fig.colorbar(cax, ax=ax3)
        
        # Set labels
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # Set ticks and labels
        ax3.set_xticks(range(len(class_names)))
        ax3.set_yticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
                        
        # Add precision/recall bar chart in the bottom row
        x = np.arange(len(class_names))
        width = 0.35
        
        precision_bars = ax4.bar(x - width/2, class_precision, width, label='Precision')
        recall_bars = ax4.bar(x + width/2, class_recall, width, label='Recall')
        
        ax4.set_title('Precision & Recall by Class')
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.set_ylim(0, 1.0)
        ax4.legend()
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on bars
        for bar, value in zip(precision_bars, class_precision):
            ax4.text(bar.get_x() + bar.get_width()/2., value + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
                    
        for bar, value in zip(recall_bars, class_recall):
            ax4.text(bar.get_x() + bar.get_width()/2., value + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_to_image(fig):
    """Converts a Matplotlib figure to a PNG image tensor and releases memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')  # Save figure
    buf.seek(0)

    # Convert PNG buffer to a TF image tensor
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.expand_dims(image, 0)  # Add batch dimension

    # **Close figure and free memory**
    plt.close(fig)
    buf.close()

    return image


# The input shape of the Wavenet model
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']))
user_checkpoint_paths = {}

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

    # Set up checkpoint path
    user_checkpoint_dir = experiment_save_path
    #os.makedirs(user_checkpoint_dir, exist_ok=True)
    user_checkpoint_path = os.path.join(user_checkpoint_dir, "ckpt")
    user_checkpoint_paths[user_test] = user_checkpoint_path
    
    # A path to log directory for Tensorboard
    log_dir = f"logs/wave_custom/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{user_test}-{run_name}"
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
                                                              users_per_class=data_parameters['validation_set'],
                                                              manual_val_dict=None)
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

    folded_weights_for_user = swimming_data.compute_folded_weights_per_user(train_dict=train_dict)
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
                                              compute_sample_weights=training_parameters['compute_sample_weights'],
                                              global_swim_weights=training_style_class_weights,
                                              global_stroke_weights=training_stroke_class_weights,
                                              folded_weights_for_user=folded_weights_for_user)


    # Get the validation data with weights and mask
    x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask = swimming_data.get_windows_dict(
        val_dict, return_weights=True, return_mask=True, transition_label=0
    )

    #val_sample_weights = utils.smooth_folded_weights_array(val_sample_weights, method='log1p', normalize=False)


    final_swim_weights = val_sample_weights * validation_style_sample_weights
    final_swim_weights = final_swim_weights / np.mean(final_swim_weights)
    final_stroke_weights = val_sample_weights[:, np.newaxis, np.newaxis] * validation_stroke_sample_weights

    if data_parameters['debug']:
        plt.hist(final_swim_weights.flatten(), bins=50)
        plt.title("Distribution of Combined Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.hist(final_stroke_weights.flatten(), bins=50)
        plt.title("Distribution of Stroke Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.hist(val_sample_weights.flatten(), bins=50)
        plt.title("Distribution of Sampled Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()


    model = build_wavenet_model(input_shape, num_styles=5, nfilt=32, output_bias=training_parameters['output_bias'])
    model.summary()



    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        validation_data = (x_val, {'swim_style_output': y_val_cat, 'stroke_label_output': y_stroke_val},
                                {'swim_style_output': validation_style_sample_weights, 'stroke_label_output': (val_stroke_mask 
                                    if training_parameters['stroke_mask'] 
                                    else validation_stroke_sample_weights)})

    elif training_parameters['swim_style_output']:
        validation_data = (x_val, y_val_cat, final_swim_weights)
      #  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_categorical_accuracy', patience=10, restore_best_weights=True, mode='max')]
       # callbacks.append(utils_train.get_callbacks(experiment_save_path, user_test, log_dir))
    else:
        validation_data = (x_val,  y_stroke_val, (val_stroke_mask if training_parameters['stroke_mask'] else final_stroke_weights))
      #  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=10, restore_best_weights=True, mode='max')]
       # callbacks.append(utils_train.get_callbacks(experiment_save_path, user_test, log_dir))
    


    # Train model
    history = train_multitask_model(
        model=model,
        train_gen=gen,  # Your generator from swimming_data.batch_generator_dicts
        val_data=validation_data,
        training_parameters=training_parameters,
        log_dir=log_dir,
        experiment_save_path=experiment_save_path,
        checkpoint_path=user_checkpoint_path
    )

    # Plot and save training history
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(experiment_save_path, "training_history.png"))
    plt.close(history_fig)

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
        pickle.dump([history], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)

    with open(os.path.join(experiment_save_path, 'training_parameters.pkl'), 'wb') as f:
        pickle.dump([training_parameters], f)
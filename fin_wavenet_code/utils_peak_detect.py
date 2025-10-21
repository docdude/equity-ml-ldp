import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

class PeakConsistencyMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val_stroke, target_threshold=0.8, consistency_weight=0.3):
        self.x_val = x_val
        self.y_val_stroke = y_val_stroke
        self.target_threshold = target_threshold
        self.consistency_weight = consistency_weight
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        stroke_pred = y_pred[1].squeeze(-1)  # Shape: (batch, time)
        stroke_true = self.y_val_stroke.squeeze(-1)
        
        # Find stroke peak centers only
        predicted_peaks = []
        
        for batch_idx in range(len(stroke_true)):
            # Find peak centers using high threshold (≥0.95 captures only center)
            true_stroke_centers = np.where(stroke_true[batch_idx] >= 0.95)[0]
            
            for stroke_pos in true_stroke_centers:
                # Get the predicted value at true stroke center
                pred_value = stroke_pred[batch_idx, stroke_pos]
                predicted_peaks.append(pred_value)
        
        if len(predicted_peaks) > 0:
            predicted_peaks = np.array(predicted_peaks)
            
            # 1. Peak height consistency (lower std = more consistent)
            peak_std = np.std(predicted_peaks)
            consistency_score = max(0, 1.0 - (peak_std / 0.2))  # Penalty for std > 0.2
            
            # 2. Fraction of stroke centers that exceed target threshold
            above_threshold = np.mean(predicted_peaks >= self.target_threshold)
            
            # 3. F1 at target threshold (using all positions, not just centers)
            stroke_pred_binary = (stroke_pred >= self.target_threshold).astype(int)
            stroke_true_binary = (stroke_true > 0.5).astype(int)
            
            y_true_flat = stroke_true_binary.flatten()
            y_pred_flat = stroke_pred_binary.flatten()
            
            if np.sum(y_true_flat) > 0:
                f1_at_threshold = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
            else:
                f1_at_threshold = 0.0
            
            # Combined objective
            peak_consistency_objective = (
                (1 - self.consistency_weight) * f1_at_threshold +
                self.consistency_weight * (0.5 * consistency_score + 0.5 * above_threshold)
            )
        else:
            peak_consistency_objective = 0.0
            consistency_score = 0.0
            above_threshold = 0.0
            f1_at_threshold = 0.0
            peak_std = 1.0
        
        # Log metrics
        logs['val_peak_consistency_objective'] = peak_consistency_objective
        logs['val_peak_std'] = peak_std
        logs['val_peaks_above_threshold'] = above_threshold
        logs['val_f1_at_target_threshold'] = f1_at_threshold
        logs['val_mean_peak_height'] = np.mean(predicted_peaks) if len(predicted_peaks) > 0 else 0.0
        logs['val_num_stroke_centers'] = len(predicted_peaks)


class EnhancedPeakConsistencyCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val_stroke, target_threshold=0.8, consistency_weight=0.3):
        self.x_val = x_val
        self.y_val_stroke = y_val_stroke
        self.target_threshold = target_threshold
        self.consistency_weight = consistency_weight
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        y_pred = self.model.predict(self.x_val, verbose=0)
        stroke_pred = y_pred[1].squeeze(-1)
        stroke_true = self.y_val_stroke.squeeze(-1)
        
        # 1. Original peak consistency metrics
        predicted_peaks = []
        for batch_idx in range(len(stroke_true)):
            true_stroke_centers = np.where(stroke_true[batch_idx] >= 0.95)[0]
            for stroke_pos in true_stroke_centers:
                pred_value = stroke_pred[batch_idx, stroke_pos]
                predicted_peaks.append(pred_value)
        
        if len(predicted_peaks) > 0:
            predicted_peaks = np.array(predicted_peaks)
            peak_std = np.std(predicted_peaks)
            consistency_score = max(0, 1.0 - (peak_std / 0.2))
            above_threshold = np.mean(predicted_peaks >= self.target_threshold)
            mean_peak_height = np.mean(predicted_peaks)
            
            # 2. False positive evaluation
            false_positives = self.detect_false_positive_peaks(stroke_pred, stroke_true, self.target_threshold)
            false_positive_penalty = np.mean(false_positives) if len(false_positives) > 0 else 0.0
            
            # 3. Peak sharpness evaluation
            sharpness_scores = self.evaluate_peak_sharpness(stroke_pred, stroke_true)
            mean_sharpness = np.mean(sharpness_scores) if len(sharpness_scores) > 0 else 0.0
            
            # 4. Threshold robustness
            threshold_metrics = self.evaluate_threshold_robustness(stroke_pred, stroke_true)
            best_f1 = max([m['f1'] for m in threshold_metrics.values()]) if threshold_metrics else 0.0
            
            # 5. Enhanced combined objective
            stroke_detection_quality = (
                0.3 * above_threshold +                    # Peak height adequacy
                0.2 * consistency_score +                  # Peak height consistency  
                0.2 * best_f1 +                           # Best achievable F1
                0.15 * min(1.0, mean_sharpness * 2) +     # Peak sharpness (scaled)
                0.15 * max(0, 1.0 - false_positive_penalty * 5)  # False positive penalty
            )
            
            # Additional practical metrics
            optimal_threshold = self.find_optimal_threshold(threshold_metrics)
            
        else:
            stroke_detection_quality = 0.0
            consistency_score = 0.0
            above_threshold = 0.0
            mean_peak_height = 0.0
            false_positive_penalty = 1.0
            mean_sharpness = 0.0
            best_f1 = 0.0
            optimal_threshold = 0.5
        
        # Log comprehensive metrics
        logs['val_stroke_detection_quality'] = stroke_detection_quality
        logs['val_peak_consistency'] = consistency_score
        logs['val_peaks_above_threshold'] = above_threshold
        logs['val_mean_peak_height'] = mean_peak_height
        logs['val_false_positive_penalty'] = false_positive_penalty
        logs['val_peak_sharpness'] = mean_sharpness
        logs['val_best_f1_across_thresholds'] = best_f1
        logs['val_optimal_threshold'] = optimal_threshold
        logs['val_num_stroke_centers'] = len(predicted_peaks) if len(predicted_peaks) > 0 else 0
    
    def find_optimal_threshold(self, threshold_metrics):
        if not threshold_metrics:
            return 0.5
        return max(threshold_metrics.keys(), key=lambda k: threshold_metrics[k]['f1'])

    def detect_false_positive_peaks(self, stroke_pred, stroke_true, peak_threshold=0.6):
        """Detect peaks in regions where no stroke should occur"""
        false_positive_peaks = []
        
        for batch_idx in range(len(stroke_true)):
            # Find regions with no stroke activity (true value < 0.1)
            non_stroke_regions = stroke_true[batch_idx] < 0.05
            
            # Find high predictions in these regions
            false_peaks = np.where((stroke_pred[batch_idx] > peak_threshold) & non_stroke_regions)[0]
            false_positive_peaks.extend(stroke_pred[batch_idx][false_peaks])
        
        return np.array(false_positive_peaks)

    def evaluate_peak_sharpness(self, stroke_pred, stroke_true, center_threshold=0.95):
        """Evaluate how sharp/narrow the predicted peaks are"""
        sharpness_scores = []
        
        for batch_idx in range(len(stroke_true)):
            true_centers = np.where(stroke_true[batch_idx] >= center_threshold)[0]
            
            for center in true_centers:
                # Look at ±3 samples around center (similar to your sigma=2.0 kernel)
                start_idx = max(0, center - 3)
                end_idx = min(len(stroke_pred[batch_idx]), center + 4)
                
                local_pred = stroke_pred[batch_idx][start_idx:end_idx]
                center_val = stroke_pred[batch_idx][center]
                
                # Sharp peak should have center >> surrounding values
                if len(local_pred) > 1:
                    surrounding_mean = np.mean(np.concatenate([local_pred[:len(local_pred)//2], 
                                                            local_pred[len(local_pred)//2+1:]]))
                    sharpness = center_val - surrounding_mean if center_val > surrounding_mean else 0
                    sharpness_scores.append(sharpness)
        
        return np.array(sharpness_scores)
        
    def evaluate_threshold_robustness(self, stroke_pred, stroke_true, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """Find the threshold that gives best F1 and most consistent performance"""
        threshold_metrics = {}
        
        stroke_true_binary = (stroke_true > 0.5).astype(int)
        
        for thresh in thresholds:
            stroke_pred_binary = (stroke_pred >= thresh).astype(int)
            
            y_true_flat = stroke_true_binary.flatten()
            y_pred_flat = stroke_pred_binary.flatten()
            
            if np.sum(y_true_flat) > 0:
                f1 = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
                precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
                recall = recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
                
                threshold_metrics[thresh] = {
                    'f1': f1, 'precision': precision, 'recall': recall
                }
        
        return threshold_metrics


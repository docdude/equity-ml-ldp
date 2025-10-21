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
    def __init__(self, x_val, y_val_stroke, target_threshold=0.8):
        self.x_val = x_val
        self.y_val_stroke = y_val_stroke
        self.target_threshold = target_threshold
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        y_pred = self.model.predict(self.x_val, verbose=0)
        stroke_pred = y_pred[1].squeeze(-1)
        stroke_true = self.y_val_stroke.squeeze(-1)

        # Count ALL predictions above threshold (not just at true positions)
        total_predicted_strokes = np.sum(stroke_pred >= self.target_threshold)
        total_true_strokes = np.sum(stroke_true >= 0.95)
        
        predicted_to_true_ratio = total_predicted_strokes / total_true_strokes if total_true_strokes > 0 else 0
        logs['val_predicted_to_true_ratio'] = predicted_to_true_ratio
        
        # 1. Original peak consistency metrics
        predicted_peaks = []
        for batch_idx in range(len(stroke_true)):
            true_stroke_centers = np.where(stroke_true[batch_idx] >= 0.95)[0]
            for stroke_pos in true_stroke_centers:
                pred_value = stroke_pred[batch_idx, stroke_pos]
                predicted_peaks.append(pred_value)
        
        if len(predicted_peaks) > 0:
            predicted_peaks = np.array(predicted_peaks)

            peaks_above_threshold = np.mean(predicted_peaks >= self.target_threshold)

            mean_peak = np.mean(predicted_peaks)
            std_peak = np.std(predicted_peaks)
            if mean_peak > 0.1:
                peak_consistency = 1.0 - (std_peak /mean_peak)
            else:
                peak_consistency = 0.0

            single_thresh_metrics = self.calculate_single_threshold_performance(
                stroke_pred, stroke_true, self.target_threshold
            )
            threshold_stability, f1_scores = self.check_threshold_stability(
                stroke_pred, stroke_true
            )
            
            # 5. False positive rate (critical with imbalanced data)
            fpr = single_thresh_metrics['fpr']
            
            # 3. Peak sharpness evaluation
            sharpness_scores = self.evaluate_peak_sharpness(stroke_pred, stroke_true)
            mean_peak_sharpness = np.mean(sharpness_scores) if len(sharpness_scores) > 0 else 0.0
            
            # Combined objective
            val_single_threshold_objective = (
                0.3 * single_thresh_metrics['f1'] +      # Overall performance
                0.25 * peaks_above_threshold +               # Peak heights adequate
                0.15 * peak_consistency +                 # Consistent heights
                0.1 * threshold_stability +             # Performance across thresholds
                0.1 * mean_peak_sharpness +             # Peak sharpness
                0.1 * max(0, 1.0 - fpr * 2)            # Low false positive rate
            )
            
            # Additional metrics for detailed tracking
            logs['val_single_threshold_objective'] = val_single_threshold_objective
            logs['val_peaks_above_target'] = peaks_above_threshold
            logs['val_peak_consistency'] = peak_consistency
            logs['val_threshold_stability'] = threshold_stability
            logs['val_peak_sharpness'] = mean_peak_sharpness
            logs['val_false_positive_rate'] = fpr
            logs['val_single_threshold_f1'] = single_thresh_metrics['f1']
            logs['val_f1_scores_across_thresholds'] = np.mean(f1_scores)
        else:
            # Handle case with no peaks
            logs['val_single_threshold_objective'] = 0.0
            logs['val_peaks_above_target'] = 0.0
            logs['val_peak_consistency'] = 0.0
            logs['val_threshold_stability'] = 0.0
            logs['val_peak_sharpness'] = 0.0
            logs['val_false_positive_rate'] = 1.0
            logs['val_single_threshold_f1'] = 0.0
            logs['val_f1_scores_across_thresholds'] = 0.0

    def evaluate_peak_sharpness(self, stroke_pred, stroke_true, center_threshold=0.95):
        peak_quality_scores = []
        
        for batch_idx in range(len(stroke_true)):
            true_centers = np.where(stroke_true[batch_idx] >= center_threshold)[0]
            
            for center in true_centers:
                window_start = max(0, center - 6)
                window_end = min(len(stroke_pred[batch_idx]), center + 7)
                local_pred = stroke_pred[batch_idx][window_start:window_end]
                center_idx = center - window_start
                
                if center_idx < len(local_pred):
                    center_val = local_pred[center_idx]
                    
                    # Count values above key thresholds based on your kernel
                    above_08 = np.sum(local_pred > 0.8)   # Should be ~3 (center + ±1)
                    above_06 = np.sum(local_pred > 0.6)   # Should be ~5 (center + ±2)
                    
                    # Calculate quality scores
                    height_score = min(center_val / 0.85, 1.0)
                    
                    # Ideal peak matches your sigma=2.0 kernel shape
                    width_08_score = 1.0 if above_08 <= 3 else max(0, 1.0 - (above_08 - 3) / 5)
                    width_06_score = 1.0 if above_06 <= 5 else max(0, 1.0 - (above_06 - 5) / 7)
                    
                    # Combine scores
                    quality = (
                        0.4 * height_score +      # Peak reaches target height
                        0.3 * width_08_score +    # Narrow at 0.8 threshold  
                        0.3 * width_06_score      # Appropriate width at 0.6 threshold
                    )
                    
                    peak_quality_scores.append(quality)
        
        return np.array(peak_quality_scores)


    def calculate_single_threshold_performance(self, stroke_pred, stroke_true, threshold=0.75):
        """Evaluate how well a single threshold works across all data"""
        
        # Binary ground truth (any part of Gaussian heatmap)
        stroke_true_binary = (stroke_true >= 0.95).astype(int)
        
        # Apply single threshold
        stroke_pred_binary = (stroke_pred >= threshold).astype(int)
        
        # Flatten for metrics
        y_true_flat = stroke_true_binary.flatten()
        y_pred_flat = stroke_pred_binary.flatten()
        
        # Calculate metrics
        if np.sum(y_true_flat) > 0:
            tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
            fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
            fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
            tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # False positive rate (critical with 98% negative class)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            return {'f1': f1, 'precision': precision, 'recall': recall, 'fpr': fpr}
        
        return {'f1': 0, 'precision': 0, 'recall': 0, 'fpr': 1}

    def check_threshold_stability(self, stroke_pred, stroke_true, threshold_range=[0.65, 0.85]):
        """Check if performance is stable across a range of thresholds"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 9)
        f1_scores = []
        
        for thresh in thresholds:
            metrics = self.calculate_single_threshold_performance(stroke_pred, stroke_true, thresh)
            f1_scores.append(metrics['f1'])
        
        # Low std means stable performance across threshold range
        threshold_stability = 1.0 - np.std(f1_scores)
        return threshold_stability, f1_scores

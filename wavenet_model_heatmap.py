"""
WaveNet Model with Barrier Heatmap Outputs (Regression)
========================================================

Adapted from swimming stroke detection using Gaussian heatmaps.

Instead of 3-class classification {DOWN, TIMEOUT, UP}, predicts:
- 2 channels: [SL_probability, TP_probability]
- Continuous [0, 1] values using sigmoid activation
- Simple thresholding for signals (no peak detection needed)

Key advantages:
1. No TIMEOUT class to overpredict
2. Smooth gradients (MSE/BCE loss)
3. Natural confidence (magnitude = certainty)
4. Handles multiple barrier touches in window
5. Temporal flexibility (doesn't need exact timing)

Based on: Swimming stroke heatmap approach (fin_wavenet_code/)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, BatchNormalization,
    GlobalAveragePooling1D, Multiply, Add, Activation
)
from tensorflow.keras.models import Model
import numpy as np


def wavenet_residual_block(x, filters, dilation_rate, block_id):
    """
    WaveNet residual block with dilated causal convolutions.
    
    Args:
        x: Input tensor
        filters: Number of filters
        dilation_rate: Dilation rate for temporal receptive field
        block_id: Block identifier for naming
    
    Returns:
        residual, skip: Residual and skip connections
    """
    # Dilated causal convolution
    tanh_out = Conv1D(
        filters, 
        kernel_size=2,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='tanh',
        name=f'block{block_id}_dilated_conv_tanh'
    )(x)
    
    sigmoid_out = Conv1D(
        filters,
        kernel_size=2,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='sigmoid',
        name=f'block{block_id}_dilated_conv_sigmoid'
    )(x)
    
    # Gated activation
    gated = Multiply(name=f'block{block_id}_gated')([tanh_out, sigmoid_out])
    
    # 1x1 convolutions for residual and skip
    residual = Conv1D(
        filters,
        kernel_size=1,
        name=f'block{block_id}_residual'
    )(gated)
    
    skip = Conv1D(
        filters,
        kernel_size=1,
        name=f'block{block_id}_skip'
    )(gated)
    
    # Add residual connection
    residual = Add(name=f'block{block_id}_add')([x, residual])
    
    return residual, skip


def deep_wavenet(inputs, filters=32, n_blocks=4):
    """
    Deep WaveNet encoder with stacked residual blocks.
    
    Builds temporal receptive field through dilated convolutions:
    - Block 1: dilation=1 (sees 2 timesteps)
    - Block 2: dilation=2 (sees 4 timesteps)
    - Block 3: dilation=4 (sees 8 timesteps)
    - Block 4: dilation=8 (sees 16 timesteps)
    
    Total receptive field: ~30 timesteps for n_blocks=4
    """
    # Initial projection
    x = Conv1D(filters, kernel_size=1, name='initial_conv')(inputs)
    
    skip_connections = []
    
    # Stack WaveNet blocks with exponentially increasing dilation
    for block in range(n_blocks):
        dilation_rate = 2 ** block
        x, skip = wavenet_residual_block(x, filters, dilation_rate, block)
        skip_connections.append(skip)
    
    # Combine all skip connections
    skip_sum = Add(name='skip_sum')(skip_connections)
    
    # Post-processing
    x = Activation('relu', name='skip_relu')(skip_sum)
    x = Conv1D(filters, kernel_size=1, name='post_conv1')(x)
    x = Activation('relu', name='post_relu')(x)
    x = Conv1D(filters, kernel_size=1, name='post_conv2')(x)
    
    return x


def build_heatmap_head(features, n_channels=2, dropout_rate=0.3):
    """
    Barrier heatmap prediction head (regression).
    
    Predicts continuous [0, 1] probabilities for each timestep:
    - Channel 0: P(Stop-Loss barrier hit soon)
    - Channel 1: P(Take-Profit barrier hit soon)
    
    Similar to stroke detection: each timestep gets probability
    
    Args:
        features: Encoder features (batch, timesteps, filters)
        n_channels: Number of output channels (default: 2 for SL/TP)
        dropout_rate: Dropout for regularization
    
    Returns:
        heatmap: (batch, timesteps, n_channels) with sigmoid activation
    """
    # Optional temporal smoothing
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='heatmap_conv1')(features)
    x = Dropout(dropout_rate, name='heatmap_dropout1')(x)
    
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu', name='heatmap_conv2')(x)
    x = Dropout(dropout_rate, name='heatmap_dropout2')(x)
    
    # Output: (batch, timesteps, 2) with sigmoid [0, 1]
    heatmap = Conv1D(
        n_channels,
        kernel_size=1,
        activation='sigmoid',  # [0, 1] probability per timestep
        name='barrier_heatmap'
    )(x)
    
    return heatmap


def build_wavenet_heatmap_model(
    input_shape=(20, 25),  # (seq_len, n_features)
    wavenet_filters=32,
    wavenet_blocks=4,
    dropout_rate=0.3,
    n_channels=2  # [SL, TP]
):
    """
    Full WaveNet model with barrier heatmap outputs.
    
    Architecture:
    1. WaveNet encoder: Learns temporal patterns
    2. Heatmap head: Predicts [SL_prob, TP_prob] per timestep
    
    Training:
    - Loss: MSE or BCE on Gaussian-smoothed barrier events
    - Targets: (batch, timesteps, 2) continuous [0, 1]
    
    Inference:
    - Simple thresholding: if TP_prob[-1] > 0.5 and TP_prob > SL_prob: LONG
    - No peak detection needed (unlike exact stroke localization)
    
    Args:
        input_shape: (seq_len, n_features)
        wavenet_filters: WaveNet hidden dimension
        wavenet_blocks: Number of WaveNet blocks (receptive field)
        dropout_rate: Dropout probability
        n_channels: Output channels (2 for SL/TP)
    
    Returns:
        Keras Model with heatmap output
    """
    inputs = Input(shape=input_shape, name='input')
    
    # WaveNet encoder
    features = deep_wavenet(inputs, filters=wavenet_filters, n_blocks=wavenet_blocks)
    
    # Heatmap prediction head (like stroke detection)
    heatmap_output = build_heatmap_head(features, n_channels=n_channels, dropout_rate=dropout_rate)
    
    model = Model(inputs=inputs, outputs=heatmap_output, name='wavenet_heatmap')
    
    return model


def create_barrier_heatmap_targets(events_df, seq_len=20, sigma=2.0):
    """
    Create Gaussian heatmap targets from barrier events.
    
    Adapted from swimming stroke heatmap generation.
    
    Args:
        events_df: DataFrame with barrier events
                   Must have columns: ['t1', 'pt', 'sl'] (barrier touch times/types)
        seq_len: Sequence length (timesteps in window)
        sigma: Gaussian smoothing parameter (days)
               sigma=2.0 means ~95% of weight within ±4 days
    
    Returns:
        heatmaps: (n_samples, seq_len, 2) array
                  [:, :, 0] = SL heatmap
                  [:, :, 1] = TP heatmap
    
    Example:
        If TP hit at position 5 in 20-step window:
        - TP channel: Gaussian centered at 5 with sigma=2
        - SL channel: all zeros
    """
    n_samples = len(events_df)
    heatmaps = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    
    # Build Gaussian kernel
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
    for i, event in events_df.iterrows():
        # Determine barrier type and position
        # This needs to be adapted based on your events structure
        # Example: if event hit TP at relative position in window
        
        # Placeholder logic - adapt to your data structure:
        # barrier_type = event['type']  # 'TP' or 'SL'
        # position = event['relative_position']  # 0 to seq_len-1
        
        # Apply Gaussian at position
        # if barrier_type == 'TP':
        #     channel = 1
        # elif barrier_type == 'SL':
        #     channel = 0
        # else:
        #     continue  # TIMEOUT - both channels stay 0
        
        # for t in range(seq_len):
        #     distance = abs(t - position)
        #     if distance <= radius:
        #         heatmaps[i, t, channel] = kernel[distance + radius]
        
        pass  # Implement based on your event structure
    
    return heatmaps


def predict_signals_from_heatmap(
    heatmap_predictions,
    sl_threshold=0.5,
    tp_threshold=0.5,
    confidence_gap=0.1
):
    """
    Convert heatmap predictions to trading signals.
    
    Simple thresholding (no peak detection needed).
    
    Args:
        heatmap_predictions: (n_samples, seq_len, 2) array
                             [:, :, 0] = SL probabilities
                             [:, :, 1] = TP probabilities
        sl_threshold: Minimum prob to consider SL signal
        tp_threshold: Minimum prob to consider TP signal
        confidence_gap: Required difference between SL and TP
    
    Returns:
        signals: (n_samples,) array with {-1, 0, 1}
                 -1 = SHORT (SL likely)
                  0 = NEUTRAL (neither clear)
                 +1 = LONG (TP likely)
    
    Logic (using last timestep as prediction point):
        - If TP_prob > tp_threshold AND TP_prob > SL_prob + gap: LONG
        - If SL_prob > sl_threshold AND SL_prob > TP_prob + gap: SHORT
        - Otherwise: NEUTRAL
    """
    n_samples = len(heatmap_predictions)
    signals = np.zeros(n_samples, dtype=np.int32)
    
    # Use last timestep as prediction (or could average over window)
    sl_probs = heatmap_predictions[:, -1, 0]  # Last timestep, SL channel
    tp_probs = heatmap_predictions[:, -1, 1]  # Last timestep, TP channel
    
    # LONG signals: TP clearly higher than SL
    long_mask = (tp_probs > tp_threshold) & (tp_probs > sl_probs + confidence_gap)
    signals[long_mask] = 1
    
    # SHORT signals: SL clearly higher than TP
    short_mask = (sl_probs > sl_threshold) & (sl_probs > tp_probs + confidence_gap)
    signals[short_mask] = -1
    
    # Everything else: NEUTRAL (implicit, already 0)
    
    return signals


if __name__ == "__main__":
    print("="*80)
    print("WAVENET HEATMAP MODEL - Barrier Prediction via Regression")
    print("="*80)
    
    # Build model
    model = build_wavenet_heatmap_model(
        input_shape=(20, 25),  # 20 timesteps, 25 features
        wavenet_filters=32,
        wavenet_blocks=4,
        dropout_rate=0.3,
        n_channels=2  # [SL, TP]
    )
    
    model.summary()
    
    print("\n" + "="*80)
    print("MODEL COMPILATION")
    print("="*80)
    
    # Compile with MSE (continuous targets) or BCE (binary-like targets)
    model.compile(
        optimizer='adam',
        loss='mse',  # or 'binary_crossentropy'
        metrics=['mae', 'mse']
    )
    
    print("\n✅ Model ready for training!")
    print("\nTraining requirements:")
    print("  - Input: (batch, 20, 25) feature sequences")
    print("  - Target: (batch, 20, 2) heatmaps [SL, TP]")
    print("  - Loss: MSE or BCE on continuous [0, 1] values")
    print("\nInference:")
    print("  - Output: (batch, 20, 2) probability heatmaps")
    print("  - Signals: Simple thresholding (no peak detection)")
    print("  - Benefits: No TIMEOUT bias, smooth gradients")
    
    # Test prediction shape
    dummy_input = tf.random.normal((32, 20, 25))
    dummy_output = model(dummy_input)
    print(f"\nTest prediction shape: {dummy_output.shape}")
    print(f"  Expected: (32, 20, 2)")
    print(f"  ✅ Matches!" if dummy_output.shape == (32, 20, 2) else "  ❌ Shape mismatch!")

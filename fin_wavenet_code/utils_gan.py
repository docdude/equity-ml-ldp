import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np
from scipy import signal


def soft_categorical_loss(y_true, y_pred):
    """
    KL Divergence Loss for soft labels.
    """
    #y_true = tf.keras.backend.clip(y_true, 1e-7, 1)  # Prevent log(0)
    #y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
    return tf.keras.losses.KLDivergence()(y_true, y_pred)

# Add to discriminator loss calculation:
def compute_gradient_penalty(real_samples, fake_samples, discriminator):
    # Cast to float32 to resolve dtype mismatch
    real_samples = tf.cast(real_samples, tf.float32)
    fake_samples = tf.cast(fake_samples, tf.float32)
    
    # Compute interpolated samples
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1], 0., 1., dtype=tf.float32)
    interpolated = alpha * real_samples + (1. - alpha) * fake_samples
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    gradients = gp_tape.gradient(pred, interpolated)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-8)
    return tf.reduce_mean((gradients_norm - 1.0) ** 2)

def gradient_penalty(real_samples, fake_samples, discriminator):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = discriminator(interpolates)
    grads = tape.gradient(pred, [interpolates])[0]
    penalty = tf.reduce_mean((tf.norm(grads, axis=[1,2]) - 1.0) ** 2)
    return penalty


def compute_batch_class_weights(batch_swim_styles):
    batch_swim_styles_flat = batch_swim_styles.flatten().tolist()
    unique_classes = np.unique(batch_swim_styles_flat)

    # Compute weights per class
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=batch_swim_styles_flat)

    # Create a dictionary mapping the unique class labels to their computed weights
    class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Convert the dictionary into a tensor, ensuring it can be indexed correctly
    max_class = max(unique_classes)  # Ensure the tensor can be indexed safely
    class_weights_tensor = tf.constant(
        [class_weight_dict.get(i, 1.0) for i in range(max_class + 1)],  # Default weight = 1.0 for missing classes
        dtype=tf.float32
    )
    return class_weights_tensor

def compute_batch_sample_weights(batch_labels, return_weights=True):
    batch_labels_flat = batch_labels.flatten()
    sample_weights = compute_sample_weight('balanced', batch_labels_flat)
    # Reshape sample weights to match the original batch label shape
    sample_weights = sample_weights.reshape(batch_labels.shape[:-1])
    #sample_weights = sample_weights.squeeze(axis=-1)
    if return_weights:
        # Convert to a TensorFlow tensor with the desired shape
        sample_weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
    else: 
        sample_weights_tensor = tf.ones(sample_weights.shape, dtype=tf.float32)
    
    # Ensure the tensor has the correct shape (64, 180)
    #sample_weights_tensor = tf.squeeze(sample_weights_tensor, axis=-1)#tf.reshape(sample_weights_tensor, batch_labels.shape)
    return sample_weights_tensor

def compute_batch_sample_weights_onehot(batch_labels, return_weights=True):
    # Convert one-hot encoded labels to class indices (64, 180)
    batch_labels_indices = np.argmax(batch_labels, axis=-1) # Shape: (64, 180)

    # Flatten for computing class-based weights
    batch_labels_flat = batch_labels_indices.flatten()

    # Compute sample weights based on class distribution
    sample_weights = compute_sample_weight('balanced', batch_labels_flat)

    # Reshape back to (64, 180) to match batch_labels
    sample_weights = sample_weights.reshape(batch_labels_indices.shape)

    if return_weights:
        sample_weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)  # Shape: (64, 180)
    else: 
        sample_weights_tensor = tf.ones(sample_weights.shape, dtype=tf.float32)

    # Expand to match one-hot labels (64, 180, 6) for element-wise multiplication
    #sample_weights_tensor = tf.expand_dims(sample_weights_tensor, axis=-1)  # Shape: (64, 180, 1)

    return sample_weights_tensor

def apply_sensor_mean(full_data):
    """
    Centers the sensor channels in full_data by subtracting the per-sample mean
    computed over the time dimension.

    Parameters:
      full_data: A Tensor of shape (batch, time, channels) where the first 6 channels
                 correspond to sensor data.

    Returns:
      full_data_centered: A Tensor where the sensor channels (first 6) are zero-centered.
      sensor_data_centered: The centered sensor data (first 6 channels).
    """
    sensor_data = full_data[..., :6]
    # Compute per-sample mean (broadcasted over time)
    sensor_mean = tf.reduce_mean(sensor_data, axis=1, keepdims=True)  # Shape: (batch, 1, 6)

    # Center the sensor data by subtracting the computed mean
    sensor_data_centered = sensor_data - sensor_mean  # Broadcasting subtraction across time steps

    # Replace the sensor data in full_data with the centered version
    full_data_centered = tf.concat([sensor_data_centered, full_data[..., 6:]], axis=-1)

    return full_data_centered, sensor_data_centered


def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3


def progressive_label_smoothing(epoch, labels, min_smooth_real=0.8, max_smooth_real=0.95, 
                                min_smooth_fake=0.05, max_smooth_fake=0.2):
    """
    Applies progressive label smoothing to both real (1s) and fake (0s) labels over epochs.

    Args:
        epoch (int): Current training epoch.
        labels (tensor): Tensor of real (1s) or fake (0s) labels.
        min_smooth_real (float): Minimum smoothing factor for real labels.
        max_smooth_real (float): Maximum smoothing factor for real labels.
        min_smooth_fake (float): Minimum smoothing factor for fake labels.
        max_smooth_fake (float): Maximum smoothing factor for fake labels.

    Returns:
        Tensor: Smoothed labels.
    """
    # Progressive smoothing factor using exponential decay
    smoothing_factor = min_smooth_real + (max_smooth_real - min_smooth_real) * (1 - np.exp(-epoch / 50))
    
    # Apply smoothing to real labels (1s) and fake labels (0s) separately
    smoothed_labels = tf.where(
        labels > 0.5,  # Real labels (1s)
        labels * smoothing_factor + (1 - smoothing_factor) * 0.5,  # Shift 1s down slightly
        labels * (1 - smoothing_factor) + (smoothing_factor * min_smooth_fake)  # Shift 0s up slightly
    )
    
    return smoothed_labels
    
def spectral_loss(real, fake):
    """Improved spectral loss implementation that preserves frequency details."""
    # Ensure inputs are float32
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)

    # Move the time dimension to the last axis for FFT processing
    real = tf.transpose(real, perm=[0, 2, 1])  # (batch, channels, time)
    fake = tf.transpose(fake, perm=[0, 2, 1])

    # Compute power spectra
    real_power = tf.abs(tf.signal.rfft(real)) ** 2
    fake_power = tf.abs(tf.signal.rfft(fake)) ** 2

    # Add small epsilon to prevent division by zero
    epsilon = 1e-6
    real_power = real_power / (tf.reduce_max(real_power, axis=-1, keepdims=True) + epsilon)
    fake_power = fake_power / (tf.reduce_max(fake_power, axis=-1, keepdims=True) + epsilon)

    # Apply frequency weighting (gradually increasing weights for higher frequencies)
    num_freqs = tf.shape(real_power)[-1]
    weight = tf.linspace(0.5, 2.0, num=num_freqs)  # Low frequencies get lower weight, high get higher

    # Compute weighted absolute difference
    spectral_loss_value = tf.reduce_mean(weight * tf.abs(real_power - fake_power))

    return spectral_loss_value

def spectral_loss_rest(real, fake, real_styles, weight_rest=0.0):
    """Spectral loss with direct per-timestep masking for rest periods."""

    # Ensure inputs are float32
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    real_styles = tf.cast(real_styles, tf.float32)

    # Move time dimension to the last axis for FFT processing
    real = tf.transpose(real, perm=[0, 2, 1])  # (batch, channels, time)
    fake = tf.transpose(fake, perm=[0, 2, 1])

    # Compute power spectra
    real_power = tf.abs(tf.signal.rfft(real)) ** 2
    fake_power = tf.abs(tf.signal.rfft(fake)) ** 2

    # Normalize spectra
    epsilon = 1e-6
    real_power = real_power / (tf.reduce_max(real_power, axis=-1, keepdims=True) + epsilon)
    fake_power = fake_power / (tf.reduce_max(fake_power, axis=-1, keepdims=True) + epsilon)

    # Frequency weighting (low frequencies get less weight, high get more)
    num_freqs = tf.shape(real_power)[-1]
    weight = tf.linspace(0.2, 1.5, num=num_freqs)  # Reduce importance of extreme high-frequencies

    # **Apply per-timestep style mask directly to power spectra**
    style_mask = tf.where(real_styles == 0, weight_rest, 1.0)  # Shape: (batch, 180, 1)
    style_mask = tf.transpose(style_mask, perm=[0, 2, 1])  # Align for frequency domain operations

    # Compute weighted absolute difference
    spectral_diff = weight * tf.abs(real_power - fake_power)

    # **Apply style mask directly in frequency space**
    spectral_loss_value = tf.reduce_mean(style_mask * spectral_diff)

    return spectral_loss_value

def spectral_loss_orig(real, fake):
    """More stable spectral loss implementation"""
    # Ensure inputs are float32
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    # Move the time dimension to the last axis
    real = tf.transpose(real, perm=[0,2,1])  # now shape (batch=64, channels=6, time=180)
    fake = tf.transpose(fake, perm=[0,2,1])

    # Now rfft will be applied over the last axis (time=180)
    real_power = tf.abs(tf.signal.rfft(real)) ** 2
    fake_power = tf.abs(tf.signal.rfft(fake)) ** 2

    # Add small epsilon to prevent numerical instability
    epsilon = 1e-6
    real_power = real_power + epsilon
    fake_power = fake_power + epsilon
    
    # Compute log-scale difference
    log_diff = tf.math.log(real_power + 1.0) - tf.math.log(fake_power + 1.0)
    
    return tf.reduce_mean(tf.abs(log_diff))

def enhanced_spectral_loss(real_sensors, fake_sensors):
    """Enhanced spectral loss with frequency weighting that works with shape [batch, time, sensors]"""
    # Transpose to put time dimension last: [batch, sensors, time]
    real_transposed = tf.transpose(real_sensors, perm=[0, 2, 1])  # Now [64, 6, 180]
    fake_transposed = tf.transpose(fake_sensors, perm=[0, 2, 1])  # Now [64, 6, 180]
    
    # Apply FFT on the time dimension (now the last dimension)
    real_fft = tf.signal.rfft(real_transposed)  # Shape: [64, 6, 91] (180//2 + 1 = 91)
    fake_fft = tf.signal.rfft(fake_transposed)  # Shape: [64, 6, 91]
    
    # Get magnitudes with log compression
    real_mag = tf.math.log1p(tf.abs(real_fft) + 1e-6)
    fake_mag = tf.math.log1p(tf.abs(fake_fft) + 1e-6)
    
    # Weight by frequency importance (lower frequencies more important)
    freq_shape = tf.shape(real_mag)[-1]  # Last dimension is frequency
    weights = tf.linspace(1.0, 0.1, freq_shape)
    weights = tf.reshape(weights, [1, 1, -1])  # Shape [1, 1, freq] for broadcasting
    
    # Weighted loss
    spectral_loss = tf.reduce_mean(tf.square((real_mag - fake_mag) * weights))
    return spectral_loss


def spectral_loss_stft(y_true, y_pred, frame_length=64, frame_step=32, fft_length=128):
    """
    Compute spectral loss between real and generated IMU signals using STFT.
    """
    # Get shapes
    num_sensors = tf.shape(y_true)[2]
    
    # Initialize loss accumulators
    total_mse_loss = 0.0
    total_log_mse_loss = 0.0
    
    # Process each channel separately
    for i in range(num_sensors):
        # Extract channel i for all batches
        y_true_channel = y_true[:, :, i]  # Shape: [batch_size, seq_length]
        y_pred_channel = y_pred[:, :, i]  # Shape: [batch_size, seq_length]
        
        # Compute STFT
        stft_true = tf.signal.stft(
            y_true_channel, 
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        stft_pred = tf.signal.stft(
            y_pred_channel,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        # Process spectrograms and compute loss
        spec_true = tf.abs(stft_true)
        spec_pred = tf.abs(stft_pred)
        
        epsilon = 1e-6
        log_spec_true = tf.math.log(spec_true + epsilon)
        log_spec_pred = tf.math.log(spec_pred + epsilon)
        
        mse_loss = tf.reduce_mean(tf.square(spec_true - spec_pred))
        log_mse_loss = tf.reduce_mean(tf.square(log_spec_true - log_spec_pred))
        
        total_mse_loss += mse_loss
        total_log_mse_loss += log_mse_loss
    
    # Average and combine losses
    avg_mse_loss = total_mse_loss / tf.cast(num_sensors, tf.float32)
    avg_log_mse_loss = total_log_mse_loss / tf.cast(num_sensors, tf.float32)
    combined_loss = avg_mse_loss + 0.5 * avg_log_mse_loss
    
    return combined_loss

def frequency_weighted_spectral_loss_orig(y_true, y_pred, frame_length=64, frame_step=32, 
                          fft_length=128, norm_type='l2'):
    """
    Improved spectral loss using transposition instead of flattening.
    
    Args:
        y_true: Real IMU signals, shape [batch_size, sequence_length, num_sensors]
        y_pred: Generated IMU signals, shape [batch_size, sequence_length, num_sensors]
        frame_length: Length of each STFT frame (window size)
        frame_step: Step size between frames (hop size)
        fft_length: Length of FFT (should be >= frame_length)
        norm_type: 'l1' for mean absolute error, 'l2' for mean squared error
        
    Returns:
        Spectral loss value
    """
    num_sensors = tf.shape(y_true)[2]

    # Initialize loss accumulator
    total_loss = 0.0
    
    # Process each channel separately while maintaining batch dimension
    for i in range(num_sensors):
        # Extract channel i for all batches
        y_true_channel = y_true[:, :, i]  # Shape: [batch_size, seq_length]
        y_pred_channel = y_pred[:, :, i]  # Shape: [batch_size, seq_length]
        
        # Compute STFT
        stft_true = tf.signal.stft(
            y_true_channel, 
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        stft_pred = tf.signal.stft(
            y_pred_channel,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        # Convert to magnitude spectrogram
        spec_true = tf.abs(stft_true)
        spec_pred = tf.abs(stft_pred)
        
        # Apply log scaling
        epsilon = 1e-6
        log_spec_true = tf.math.log(spec_true + epsilon)
        log_spec_pred = tf.math.log(spec_pred + epsilon)
        
        # Calculate frequency resolution (Hz per bin)
        # Assuming 50Hz sampling rate as an example - adjust for your actual rate
        sample_rate = 30.0
        freq_resolution = sample_rate / fft_length
        
        # Create frequency weights based on swimming-specific knowledge
        num_freq_bins = fft_length // 2 + 1  # For RFFT

        # Calculate bin indices for specific frequency bands
        band1_end = int(1.0 / freq_resolution) + 1  # 0-1.0 Hz
        band2_end = int(3.0 / freq_resolution) + 1  # 1.0-3.0 Hz
        band3_end = int(5.0 / freq_resolution) + 1  # 3.0-5.0 Hz

        # Create frequency weights tensor that adapts to the FFT length
        freq_weights = tf.concat([
            tf.ones([band1_end]) * 0.62,                 # 0-1.0 Hz
            tf.ones([band2_end - band1_end]) * 1.98,     # 1.0-3.0 Hz
            tf.ones([band3_end - band2_end]) * 0.91,     # 3.0-5.0 Hz
            tf.ones([num_freq_bins - band3_end]) * 0.49  # Higher frequencies
        ], axis=0)     

        # Reshape frequency weights to match spectrogram dimensions for broadcasting
        # From [num_freq_bins] to [1, 1, num_freq_bins]
        freq_weights = tf.reshape(freq_weights, [1, 1, -1])
        
        # Now broadcast to [batch_size, num_frames, num_freq_bins]
        # This will automatically broadcast along batch and frame dimensions
        
        # Apply frequency weights to the spectral difference
        if norm_type.lower() == 'l1':
            # L1 norm with frequency weighting
            weighted_diff = freq_weights * tf.abs(log_spec_true - log_spec_pred)
            channel_loss = tf.reduce_mean(weighted_diff)
        else:
            # L2 norm with frequency weighting
            weighted_diff = freq_weights * tf.square(log_spec_true - log_spec_pred)
            channel_loss = tf.reduce_mean(weighted_diff)
        
        # Add to total loss (can apply channel-specific weights here if needed)
        # For example, accelerometer channels might be weighted differently than gyroscope
        channel_weight = 1.0  # Equal weighting for now
        total_loss += channel_weight * channel_loss
    
    # Normalize by number of channels
    avg_loss = total_loss / tf.cast(num_sensors, tf.float32)
    
    return avg_loss

def frequency_weighted_spectral_loss_old(y_true, y_pred, frame_length=64, frame_step=32, 
                                   fft_length=128, freq_weights=None, norm_type='l2'):
    """
    Vectorized version of spectral loss without loops.
    """
    batch_size = tf.shape(y_true)[0]
    num_sensors = tf.shape(y_true)[2]
    num_freq_bins = fft_length // 2 + 1
    
    # Transpose to [batch_size, channels, timesteps]
    y_true_t = tf.transpose(y_true, [0, 2, 1])
    y_pred_t = tf.transpose(y_pred, [0, 2, 1])
    
    # Reshape to process all channels at once
    y_true_r = tf.reshape(y_true_t, [-1, tf.shape(y_true)[1]])  # [batch_size*num_sensors, timesteps]
    y_pred_r = tf.reshape(y_pred_t, [-1, tf.shape(y_pred)[1]])  # [batch_size*num_sensors, timesteps]
    
    # Compute STFT for all channels at once
    stft_true = tf.signal.stft(
        y_true_r, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window
    )
    
    stft_pred = tf.signal.stft(
        y_pred_r,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window
    )
    
    # Process spectrograms
    spec_true = tf.abs(stft_true)
    spec_pred = tf.abs(stft_pred)
    
    # Apply log scaling
    epsilon = 1e-6
    log_spec_true = tf.math.log(spec_true + epsilon)
    log_spec_pred = tf.math.log(spec_pred + epsilon)
    
    # Create or process frequency weights
    if freq_weights is None:
        # Create default weights
        sample_rate = 30.0
        freq_resolution = sample_rate / fft_length
        
        band1_end = tf.cast(tf.math.floor(1.0 / freq_resolution), tf.int32)
        band2_end = tf.cast(tf.math.floor(3.0 / freq_resolution), tf.int32)
        band3_end = tf.cast(tf.math.floor(5.0 / freq_resolution), tf.int32)
        
        # Create frequency weights using TensorFlow operations
        band1_weights = tf.ones([band1_end]) * 0.62
        band2_weights = tf.ones([band2_end - band1_end]) * 1.98
        band3_weights = tf.ones([band3_end - band2_end]) * 0.91
        band4_weights = tf.ones([num_freq_bins - band3_end]) * 0.49
        
        freq_weights = tf.concat([band1_weights, band2_weights, band3_weights, band4_weights], axis=0)
    else:
        # Convert to tensor if needed
        if not isinstance(freq_weights, tf.Tensor):
            freq_weights = tf.convert_to_tensor(freq_weights, dtype=tf.float32)
        
        # Handle size mismatch using tf.cond
        freq_weights_size = tf.shape(freq_weights)[0]
        
        def resize_weights():
            # Use tf.cond to handle both cases: too short or too long
            def pad_weights():
                last_value = freq_weights[-1]
                # Create padding with the correct size
                padding_size = num_freq_bins - freq_weights_size
                padding = tf.ones([padding_size], dtype=tf.float32) * last_value
                return tf.concat([freq_weights, padding], axis=0)
            
            def truncate_weights():
                return freq_weights[:num_freq_bins]
            
            return tf.cond(
                freq_weights_size < num_freq_bins,
                pad_weights,
                truncate_weights
            )
        
        # Only resize if needed
        freq_weights = tf.cond(
            freq_weights_size != num_freq_bins,
            resize_weights,
            lambda: freq_weights
        )
    
    # Reshape for broadcasting: [1, 1, num_freq_bins]
    freq_weights = tf.reshape(freq_weights, [1, 1, num_freq_bins])
    
    # Calculate loss
    if norm_type.lower() == 'l1':
        weighted_diff = freq_weights * tf.abs(log_spec_true - log_spec_pred)
    else:
        weighted_diff = freq_weights * tf.square(log_spec_true - log_spec_pred)
    
    # Compute mean across all dimensions
    loss = tf.reduce_mean(weighted_diff)
    
    return loss

def frequency_weighted_spectral_loss(y_true, y_pred, frame_length=64, frame_step=32, 
                                   fft_length=128, freq_weights=None, norm_type='l2',
                                   scale_factor=0.1, log_compression=True):
    """
    Improved spectral loss with scaling and compression options.
    
    Args:
        y_true: Real sensor data
        y_pred: Generated sensor data
        frame_length: Length of each frame for STFT
        frame_step: Step size between frames
        fft_length: FFT length
        freq_weights: Optional frequency weighting
        norm_type: 'l1' or 'l2' norm
        scale_factor: Factor to scale down the final loss (default: 0.1)
        log_compression: Whether to apply log compression to the final loss
    """
    batch_size = tf.shape(y_true)[0]
    num_sensors = tf.shape(y_true)[2]
    num_freq_bins = fft_length // 2 + 1
    
    # Transpose to [batch_size, channels, timesteps]
    y_true_t = tf.transpose(y_true, [0, 2, 1])
    y_pred_t = tf.transpose(y_pred, [0, 2, 1])
    
    # Reshape to process all channels at once
    y_true_r = tf.reshape(y_true_t, [-1, tf.shape(y_true)[1]])
    y_pred_r = tf.reshape(y_pred_t, [-1, tf.shape(y_pred)[1]])
    
    # Normalize inputs to reduce magnitude differences
    # This helps stabilize the spectral differences
    y_true_norm = tf.nn.l2_normalize(y_true_r, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred_r, axis=1)
    
    # Compute STFT
    stft_true = tf.signal.stft(
        y_true_norm, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window
    )
    
    stft_pred = tf.signal.stft(
        y_pred_norm,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window
    )
    
    # Process spectrograms - use power spectrograms instead of magnitude
    # Power spectrograms (squared magnitude) help emphasize stronger components
    spec_true = tf.abs(stft_true)
    spec_pred = tf.abs(stft_pred)
    
    # Apply log scaling with smaller epsilon for better numerical stability
    epsilon = 1e-5
    log_spec_true = tf.math.log(spec_true + epsilon)
    log_spec_pred = tf.math.log(spec_pred + epsilon)
    
    # Create or process frequency weights
    if freq_weights is None:
        # Create default weights
        sample_rate = 30.0
        freq_resolution = sample_rate / fft_length
        
        band1_end = tf.cast(tf.math.floor(1.0 / freq_resolution), tf.int32)
        band2_end = tf.cast(tf.math.floor(3.0 / freq_resolution), tf.int32)
        band3_end = tf.cast(tf.math.floor(5.0 / freq_resolution), tf.int32)
        
        # Create frequency weights using TensorFlow operations
        # Normalize weights to sum to 1.0 to help control magnitude
        total_bins = tf.cast(num_freq_bins, tf.float32)
        
        band1_size = tf.cast(band1_end, tf.float32)
        band2_size = tf.cast(band2_end - band1_end, tf.float32)
        band3_size = tf.cast(band3_end - band2_end, tf.float32)
        band4_size = total_bins - tf.cast(band3_end, tf.float32)
        
        # Calculate normalized weights
        band1_weights = tf.ones([band1_end]) * 0.62 / total_bins * 4.0
        band2_weights = tf.ones([band2_end - band1_end]) * 1.98 / total_bins * 4.0
        band3_weights = tf.ones([band3_end - band2_end]) * 0.91 / total_bins * 4.0
        band4_weights = tf.ones([num_freq_bins - band3_end]) * 0.49 / total_bins * 4.0
        
        freq_weights = tf.concat([band1_weights, band2_weights, band3_weights, band4_weights], axis=0)
    else:
        # Convert to tensor if needed
        if not isinstance(freq_weights, tf.Tensor):
            freq_weights = tf.convert_to_tensor(freq_weights, dtype=tf.float32)
        
        # Handle size mismatch using tf.cond
        freq_weights_size = tf.shape(freq_weights)[0]
        
        def resize_weights():
            # Use tf.cond to handle both cases: too short or too long
            def pad_weights():
                last_value = freq_weights[-1]
                # Create padding with the correct size
                padding_size = num_freq_bins - freq_weights_size
                padding = tf.ones([padding_size], dtype=tf.float32) * last_value
                return tf.concat([freq_weights, padding], axis=0)
            
            def truncate_weights():
                return freq_weights[:num_freq_bins]
            
            return tf.cond(
                freq_weights_size < num_freq_bins,
                pad_weights,
                truncate_weights
            )
        
        # Only resize if needed
        freq_weights = tf.cond(
            freq_weights_size != num_freq_bins,
            resize_weights,
            lambda: freq_weights
        )
    
    # Reshape for broadcasting: [1, 1, num_freq_bins]
    freq_weights = tf.reshape(freq_weights, [1, 1, num_freq_bins])
    
    # Calculate loss with clipping to prevent extreme values
    if norm_type.lower() == 'l1':
        # L1 norm tends to be more stable for spectral differences
        weighted_diff = freq_weights * tf.abs(log_spec_true - log_spec_pred)
    else:
        # Clip difference to prevent extreme values in L2 loss
        diff = tf.clip_by_value(log_spec_true - log_spec_pred, -5.0, 5.0)
        weighted_diff = freq_weights * tf.square(diff)
    
    # Compute mean across all dimensions
    loss = tf.reduce_mean(weighted_diff)
    
    # Apply scaling and optional log compression
    scaled_loss = loss * scale_factor
    
    if log_compression:
        # Log compression further reduces large values while preserving small differences
        # Adding 1.0 ensures the loss remains positive
        final_loss = tf.math.log(scaled_loss + 1.0)
    else:
        final_loss = scaled_loss
    
    return final_loss

def comprehensive_spectral_loss_old(y_true, y_pred, window_weights=[0.33,0.34,0.33], freq_weights=None, norm_type='l2'):
    """
    Comprehensive spectral loss combining frequency weighting and multi-resolution analysis.
    
    Args:
        y_true: Real IMU signals, shape [batch_size, sequence_length, num_sensors]
        y_pred: Generated IMU signals, shape [batch_size, sequence_length, num_sensors]
        norm_type: 'l1' for mean absolute error, 'l2' for mean squared error
        
    Returns:
        Combined spectral loss value
    """
    # Default weights if none provided
    if window_weights is None:
        window_weights = {
            32: 0.33,   # Short window
            64: 0.34,  # Medium window
            128: 0.33   # Long window
        }
    # Initialize loss
    total_loss = 0.0
    
    # Compute loss for each window size
    for window_size, weight in window_weights.items():
        # Calculate appropriate fft_length (next power of 2 >= window_size)
        fft_length = 1
        while fft_length < window_size:
            fft_length *= 2
        
        # Compute spectral loss for this window size
        window_loss = frequency_weighted_spectral_loss(
            y_true, 
            y_pred, 
            frame_length=window_size, 
            frame_step=window_size//2, 
            fft_length=fft_length,  # Explicitly set fft_length
            freq_weights= freq_weights,
            norm_type=norm_type
        )
        weight = tf.cast(weight, tf.float32)
        # Add weighted loss
        total_loss += weight * window_loss
    
    return total_loss

def comprehensive_spectral_loss(y_true, y_pred, window_weights=None, freq_weights=None, 
                               norm_type='l2', scale_factor=0.1, log_compression=True):
    """
    Comprehensive spectral loss combining frequency weighting and multi-resolution analysis
    with improved scaling options.
    
    Args:
        y_true: Real IMU signals, shape [batch_size, sequence_length, num_sensors]
        y_pred: Generated IMU signals, shape [batch_size, sequence_length, num_sensors]
        window_weights: Dictionary of window sizes and their weights, or list of weights for default sizes
        freq_weights: Optional frequency weights to pass to the spectral loss function
        norm_type: 'l1' for mean absolute error, 'l2' for mean squared error
        scale_factor: Factor to scale down the final loss (default: 0.1)
        log_compression: Whether to apply log compression to the final loss
        
    Returns:
        Combined and scaled spectral loss value
    """
    # Handle window weights based on input type
    if window_weights is None:
        # Default window sizes and weights
        window_dict = {
            32: 0.33,   # Short window
            64: 0.34,   # Medium window
            128: 0.33   # Long window
        }
    elif isinstance(window_weights, list) or isinstance(window_weights, tuple):
        # If weights are provided as a list, use with default window sizes
        if len(window_weights) != 3:
            raise ValueError("When providing window_weights as a list, it must have exactly 3 elements")
        
        window_dict = {
            32: float(window_weights[0]),   # Short window
            64: float(window_weights[1]),   # Medium window
            128: float(window_weights[2])   # Long window
        }
    elif isinstance(window_weights, dict):
        # Use provided dictionary directly
        window_dict = window_weights
    else:
        raise ValueError("window_weights must be None, a list of 3 weights, or a dictionary mapping window sizes to weights")
    
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Initialize loss accumulator
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_weight = tf.constant(0.0, dtype=tf.float32)
    
    # Compute loss for each window size
    for window_size, weight in window_dict.items():
        # Cast window size to int if it's not already
        window_size = int(window_size)
        
        # Calculate appropriate fft_length (next power of 2 >= window_size)
        fft_length = 1
        while fft_length < window_size:
            fft_length *= 2
        
        # Cast weight to float32
        weight_tf = tf.cast(weight, tf.float32)
        
        # Compute spectral loss for this window size
        window_loss = frequency_weighted_spectral_loss(
            y_true, 
            y_pred, 
            frame_length=window_size, 
            frame_step=window_size//2, 
            fft_length=fft_length,
            freq_weights=freq_weights,
            norm_type=norm_type,
            scale_factor=1.0,  # No scaling at individual window level
            log_compression=False  # No compression at individual window level
        )
        
        # Add weighted loss
        total_loss += weight_tf * window_loss
        total_weight += weight_tf
    
    # Normalize by total weight
    if total_weight > 0:
        total_loss = total_loss / total_weight
    
    # Apply final scaling and compression
    scaled_loss = total_loss * tf.cast(scale_factor, tf.float32)
    
    if log_compression:
        # Log compression to reduce the impact of large values
        # Adding 1.0 ensures the loss remains positive and non-zero
        final_loss = tf.math.log(scaled_loss + 1.0)
    else:
        final_loss = scaled_loss
    
    return final_loss

def temporal_consistency_loss(generated_sensors):
    """Compute temporal smoothness loss"""
    return tf.reduce_mean(tf.abs(generated_sensors[:, 1:] - generated_sensors[:, :-1]))

def temporal_style_loss(generated_styles):
    """L2-based temporal loss to enforce smooth transitions in swim styles"""
    return tf.reduce_mean(tf.square(generated_styles[:, 1:] - generated_styles[:, :-1]))

def temporal_style_stability_loss(real_styles, fake_styles):
    """
    Penalizes style changes in generated data only where the real data is stable.
    """
    real_styles = tf.squeeze(real_styles, axis=-1)

    # Compute transitions in real and fake styles
    real_transitions = tf.abs(real_styles[:, 1:] - real_styles[:, :-1])  # Real data transitions
    fake_transitions = tf.abs(fake_styles[:, 1:] - fake_styles[:, :-1])  # Fake data transitions

    # Create a mask where real data is **constant** (i.e., no transition occurred)
    no_transition_mask = tf.cast(tf.equal(real_transitions, 0), tf.float32)  # 1 where real is stable, 0 otherwise

    # Penalize fake transitions **only where the real data was stable**
    stability_penalty = tf.reduce_mean(no_transition_mask * tf.square(fake_transitions))

    return stability_penalty

def transition_sharpness_loss(real_styles, fake_styles):
    """Encourages sharp transitions instead of gradual drifting"""

    real_styles = tf.squeeze(real_styles, axis=-1)
    fake_styles = tf.round(fake_styles)

    # Compute absolute differences between consecutive time steps
    real_transitions = tf.abs(real_styles[:, 1:] - real_styles[:, :-1])  # (batch, 179)
    fake_transitions = tf.abs(fake_styles[:, 1:] - fake_styles[:, :-1])  # (batch, 179)

    # Identify real transitions (where style changes)
    real_transition_mask = tf.cast(real_transitions > 0, tf.float32)  # (batch, 179)

    # Penalize fake styles that fail to transition sharply where real transitions occur
    transition_penalty = tf.reduce_mean(real_transition_mask * tf.square(fake_transitions - real_transitions))

    return transition_penalty

def transition_sharpness_loss_debug(real_styles, fake_styles):
    """Encourages sharp transitions instead of gradual drifting"""
    real_styles = tf.squeeze(real_styles, axis=-1)
    fake_styles = tf.round(fake_styles)

    # Compute absolute differences between consecutive time steps
    real_transitions = tf.abs(real_styles[:, 1:] - real_styles[:, :-1])  # (batch, 179)
    fake_transitions = tf.abs(fake_styles[:, 1:] - fake_styles[:, :-1])  # (batch, 179)
    
    # Identify real transitions (where style changes)
    real_transition_mask = tf.cast(real_transitions > 0, tf.bool)
    fake_transition_mask = tf.cast(fake_transitions > 0, tf.bool)
    
    # Get transition values
    real_transition_values = tf.boolean_mask(real_transitions, real_transition_mask)
    fake_at_real_values = tf.boolean_mask(fake_transitions, real_transition_mask)
    
    # Calculate statistics
    total_real_transitions = tf.reduce_sum(tf.cast(real_transition_mask, tf.int32))
    total_fake_transitions = tf.reduce_sum(tf.cast(fake_transition_mask, tf.int32))
    matching_transitions = tf.reduce_sum(
        tf.cast(tf.logical_and(real_transition_mask, fake_transition_mask), tf.int32))
    
    # Print transition analysis
    tf.cond(
        tf.greater(total_real_transitions, 0),
        lambda: tf.print(
            "\n=== Transition Analysis ===",
            "\nTotal real transitions:", total_real_transitions,
            "\nTotal fake transitions:", total_fake_transitions,
            "\nMatching transitions:", matching_transitions,
            "\nTransition match rate:", matching_transitions / total_real_transitions,
            "\nReal transition magnitudes:", real_transition_values,
            "\nFake values at real transitions:", fake_at_real_values,
            "\n==========================="
        ),
        lambda: tf.print("No transitions in this batch")
    )
    
    # Penalize fake styles that fail to transition sharply where real transitions occur
    transition_penalty = tf.reduce_mean(
        tf.cast(real_transition_mask, tf.float32) * tf.square(fake_transitions - real_transitions))

    return transition_penalty
    
def enhanced_transition_loss(real_styles, fake_styles, transition_weight=10.0):
    """Heavily weighted loss function focusing on style transitions with careful dimension handling"""
    # Ensure proper tensor shapes and types
    real_styles = tf.cast(tf.squeeze(real_styles, axis=-1), tf.float32)
    fake_styles = tf.cast(tf.round(fake_styles), tf.float32)
    
    # Get sequence length
    seq_length = tf.shape(real_styles)[1]
    
    # Compute absolute differences between consecutive time steps
    real_transitions = tf.abs(real_styles[:, 1:] - real_styles[:, :-1])  # (batch, seq_length-1)
    fake_transitions = tf.abs(fake_styles[:, 1:] - fake_styles[:, :-1])  # (batch, seq_length-1)
    
    # Identify real transitions (where style changes)
    real_transition_mask = tf.cast(real_transitions > 0, tf.float32)  # (batch, seq_length-1)
        
    # Specific transition loss (heavily weighted)
    transition_match_loss = tf.reduce_mean(
        real_transition_mask * tf.square(fake_transitions - real_transitions)
    )
    
    # For neighborhood loss, we'll use a simpler approach to avoid dimension issues
    # We'll focus on the timesteps around transitions
    
    # Create a mask for the full sequence that marks positions near transitions
    # First, create a full-length mask with transitions marked
    full_transition_mask = tf.concat([
        tf.zeros((tf.shape(real_styles)[0], 1), dtype=tf.float32),  # Add zero at the beginning
        real_transition_mask,
    ], axis=1)  # Now shape (batch, seq_length)
    
    # Create masks for positions before and after transitions
    pre_transition_mask = tf.concat([
        tf.zeros((tf.shape(real_styles)[0], 1), dtype=tf.float32),  # Add zero at the beginning
        full_transition_mask[:, :-1]  # Shift right
    ], axis=1)  # Now shape (batch, seq_length)
    
    post_transition_mask = tf.concat([
        full_transition_mask[:, 1:],  # Shift left
        tf.zeros((tf.shape(real_styles)[0], 1), dtype=tf.float32)  # Add zero at the end
    ], axis=1)  # Now shape (batch, seq_length)
    
    # Combine masks for neighborhood
    neighborhood_mask = full_transition_mask + pre_transition_mask + post_transition_mask
    neighborhood_mask = tf.minimum(neighborhood_mask, 1.0)  # Cap at 1.0
    
    # Apply neighborhood mask to style differences
    neighborhood_loss = tf.reduce_mean(
        neighborhood_mask * tf.square(real_styles - fake_styles)
    )
    
    # Combine losses with heavy weighting on transitions
    total_loss = transition_weight * transition_match_loss + 5.0 * neighborhood_loss
    
    return total_loss

def transition_focal_loss(real_styles, fake_styles, gamma=2.0):
    """Focal loss variant that focuses on hard-to-match transitions"""
    real_styles = tf.squeeze(real_styles, axis=-1)
    fake_styles = tf.round(fake_styles)
    
    # Style difference at each time step
    style_diff = tf.cast(real_styles != fake_styles, tf.float32)
    
    # Compute transitions
    real_transitions = tf.abs(real_styles[:, 1:] - real_styles[:, :-1])
    real_transition_mask = tf.cast(real_transitions > 0, tf.float32)
    
    # Calculate probability of correct prediction (inverse of error)
    p_correct = 1.0 - style_diff
    
    # Apply focal loss modulation: (1-p)^gamma * loss
    # This increases the weight for misclassified examples
    focal_weights = tf.pow(1.0 - p_correct, gamma)
    
    # Extra weight for transition points and their neighbors
    transition_weights = tf.pad(real_transition_mask, [[0, 0], [0, 1]])  # Current and next
    transition_weights = tf.maximum(transition_weights, 
                                   tf.pad(real_transition_mask, [[0, 0], [1, 0]]))  # Previous
    
    # Combine weights
    combined_weights = focal_weights * (1.0 + 5.0 * transition_weights)
    
    # Final loss
    loss = tf.reduce_mean(combined_weights * tf.square(real_styles - fake_styles))
    
    return loss

def perfect_transition_loss(real_styles, fake_styles, power=6.0, perfect_weight=500.0):
    """Loss function optimized for achieving 100% accuracy at transition points"""
    # Ensure proper tensor shapes and types
    real_styles = tf.cast(tf.squeeze(real_styles, axis=-1), tf.float32)
    fake_styles = tf.cast(fake_styles, tf.float32)
    
    # Round fake styles for error detection (but keep original for gradient flow)
    fake_styles_rounded = tf.round(fake_styles)
    
    # Find transitions in real data
    real_diffs = real_styles[:, 1:] - real_styles[:, :-1]
    real_transitions = tf.cast(tf.abs(real_diffs) > 0, tf.float32)
    
    # Create a mask that highlights transition neighborhoods (3 timesteps)
    # Current transition point
    current_mask = tf.pad(real_transitions, [[0, 0], [0, 1]])
    
    # One timestep before transition
    before_mask = tf.pad(real_transitions, [[0, 0], [1, 0]])
    
    # One timestep after transition
    after_mask = tf.pad(tf.pad(real_transitions, [[0, 0], [0, 1]]), [[0, 0], [1, 0]])[:, 1:]
    
    # Combine masks for transition neighborhood
    transition_mask = current_mask + before_mask + after_mask
    transition_mask = tf.minimum(transition_mask, 1.0)  # Cap at 1.0
    
    # Basic style matching loss for all points
    base_loss = tf.square(real_styles - fake_styles)
    
    # Exponential penalty at transition points
    # Using higher power creates extremely steep penalty for remaining errors
    transition_loss = tf.pow(tf.abs(real_styles - fake_styles), power)
    
    # Combine losses based on mask
    combined_loss = (1 - transition_mask) * base_loss + transition_mask * transition_loss
    
    # Additional perfect penalty for any rounded errors at transitions
    # This adds a large constant penalty for each remaining error
    errors_at_transitions = tf.cast(
        tf.logical_and(
            real_styles != fake_styles_rounded,
            transition_mask > 0
        ), 
        tf.float32
    )
    perfect_penalty = tf.reduce_sum(errors_at_transitions) * perfect_weight
    
    # Final loss combines mean loss with perfect penalty
    final_loss = tf.reduce_mean(combined_loss) + perfect_penalty
    
    # Print debugging info (can be removed in production)
    tf.print("Transition errors:", tf.reduce_sum(errors_at_transitions), 
             "out of", tf.reduce_sum(transition_mask), "transition points")
    
    return final_loss

def sensor_distribution_loss(real_sensors, fake_sensors):
    """Ensure generated sensor data maintains similar distribution to real data"""
    # Convert both inputs to float32
    #real_sensors = tf.cast(real_sensors, tf.float32)
    #fake_sensors = tf.cast(fake_sensors, tf.float32)
    
    real_mean = tf.reduce_mean(real_sensors, axis=[0,1])  # shape (channels,)
    fake_mean = tf.reduce_mean(fake_sensors, axis=[0,1])
    real_std  = tf.math.reduce_std(real_sensors, axis=[0,1])
    fake_std  = tf.math.reduce_std(fake_sensors, axis=[0,1])
    
    mean_loss = tf.reduce_mean(tf.abs(real_mean - fake_mean))
    std_loss = tf.reduce_mean(tf.abs(real_std - fake_std))
    zero_mean_loss = tf.reduce_mean(tf.abs(fake_mean))  # 🚨 Enforce zero-centered output
 
    return mean_loss + std_loss + zero_mean_loss

def transition_boundary_loss(real_styles, fake_styles, transition_weight=15.0):
    """Loss function specifically targeting transition boundaries"""
    # Ensure proper tensor shapes and types
    real_styles = tf.cast(tf.squeeze(real_styles, axis=-1), tf.float32)
    fake_styles = tf.cast(fake_styles, tf.float32)  # Don't round here to allow gradient flow
    
    # Basic style matching loss (using soft targets for gradient flow)
    style_match_loss = tf.reduce_mean(tf.square(real_styles - fake_styles))
    
    # Find transitions in real data
    real_diffs = real_styles[:, 1:] - real_styles[:, :-1]
    real_transitions = tf.cast(tf.abs(real_diffs) > 0, tf.float32)
    
    # If there are no transitions, return only style matching loss
    if tf.reduce_sum(real_transitions) == 0:
        return style_match_loss
    
    # Get transition indices
    transition_indices = tf.cast(tf.where(tf.abs(real_diffs) > 0), tf.int32)

    # For each transition, identify a window of timesteps around it
    window_size = 3  # Look at 3 timesteps before and after transition
    batch_size = tf.shape(real_styles)[0]
    seq_length = tf.shape(real_styles)[1]
    
    # Initialize tensor to accumulate boundary penalties
    boundary_penalties = tf.zeros_like(real_styles)
    
    # For each transition point, add increasing penalties as we get closer to the transition
    for i in range(-window_size, window_size + 1):
        # Weight higher closer to transition (0 at edges, 1 at center)
        weight = 1.0 - tf.abs(tf.cast(i, tf.float32) / (window_size + 1))
        
        # Create shifted indices (for positions around transitions)
        batch_indices = transition_indices[:, 0]
        time_indices = transition_indices[:, 1] + i
        
        # Ensure indices are within bounds
        valid_mask = (time_indices >= 0) & (time_indices < seq_length)
        valid_batch_indices = tf.boolean_mask(batch_indices, valid_mask)
        valid_time_indices = tf.boolean_mask(time_indices, valid_mask)
        
        # Create indices for scatter update
        valid_indices = tf.stack([valid_batch_indices, valid_time_indices], axis=1)
        
        # Create update values (weighted penalties)
        updates = tf.ones(tf.shape(valid_indices)[0], dtype=tf.float32) * weight
        
        # Add penalties to boundary_penalties tensor
        boundary_penalties = tf.tensor_scatter_nd_update(
            boundary_penalties, valid_indices, updates)
    
    # Apply boundary penalties to style differences
    boundary_loss = tf.reduce_sum(
        boundary_penalties * tf.square(real_styles - fake_styles)) / (
        tf.reduce_sum(boundary_penalties) + 1e-8)  # Normalize by sum of weights
    
    # Combined loss with heavy weighting on transition boundaries
    total_loss = style_match_loss + transition_weight * boundary_loss
    
    return total_loss


def sensor_distribution_loss_per_channel(real_sensors, fake_sensors):
    """Ensure generated sensor data maintains similar distribution per-channel."""
    # Convert both inputs to float32
    real_sensors = tf.cast(real_sensors, tf.float32)
    fake_sensors = tf.cast(fake_sensors, tf.float32)
    
    # Compute per-channel mean & std
    real_mean = tf.reduce_mean(real_sensors, axis=[0,1], keepdims=True)  # Shape: (1,1,6)
    fake_mean = tf.reduce_mean(fake_sensors, axis=[0,1], keepdims=True)
    real_std = tf.math.reduce_std(real_sensors, axis=[0,1], keepdims=True)
    fake_std = tf.math.reduce_std(fake_sensors, axis=[0,1], keepdims=True)

    # Compute mean and std differences for each channel
    mean_loss = tf.reduce_mean(tf.abs(real_mean - fake_mean) / (tf.abs(real_mean) + tf.abs(fake_mean) + 1e-8))
    std_loss = tf.reduce_mean(tf.abs(real_std - fake_std) / (tf.abs(real_std) + 1e-8))

    return mean_loss + std_loss  # Weight this loss in the total generator loss

def zero_motion_loss(real_styles, fake_sensors, target_style=0):
    """
    Penalise energy in the generated sensors when style == target_style (rest).

    Args
        real_styles  : (B, 180, 1)  integer labels 0-5 (or float, it’s casted)
        fake_sensors : (B, 180, 6)  generated sensor values
    Returns
        scalar loss  –  mean squared amplitude on the timesteps where style==target_style
    """
    # 1) build mask [B,T,1]
    mask = tf.cast(tf.equal(real_styles, target_style), tf.float32)

    # 2) energy per timestep  [B,T,1]
    energy = tf.reduce_mean(tf.square(fake_sensors), axis=-1, keepdims=True)

    # 3) masked average
    num = tf.reduce_sum(mask * energy)                       # Σ energy on rest steps
    den = tf.reduce_sum(mask) + 1e-8                         # number of rest steps
    return num / den

def zero_style_sensor_loss_L1L2(real_sensors,
                            fake_sensors,
                            real_styles,
                            target_style=0,
                            norm='l1'):
    """
    L{1,2} distance between real and fake sensors restricted to target_style.

    Args
        real_sensors  : (B,180,6)
        fake_sensors  : (B,180,6)
        real_styles   : (B,180,1)
        target_style  : int (usually 0 = rest)
        norm          : 'l1' | 'l2'
    """
    mask = tf.cast(tf.equal(real_styles, target_style), tf.float32)  # (B,T,1)
    if norm == 'l1':
        diff = tf.abs(real_sensors - fake_sensors)
    elif norm == 'l2':
        diff = tf.square(real_sensors - fake_sensors)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    diff *= mask                                  # broadcast along channel axis

    # correct normalisation: count *all* masked elements (time × channels)
    denom = tf.reduce_sum(mask) * tf.cast(tf.shape(real_sensors)[-1], tf.float32)
    return tf.reduce_sum(diff) / (denom + 1e-8)

def zero_style_sensor_loss(real_sensors,
                            fake_sensors,
                            real_styles,
                            target_style=0, 
                            delta=0.01):
    mask = tf.cast(tf.equal(real_styles, target_style), tf.float32)
    error = fake_sensors - real_sensors
    abs_err = tf.abs(error)
    # element-wise Huber
    quad = tf.minimum(abs_err,  delta)
    lin  = abs_err - quad
    diff = 0.5 * tf.square(quad) + delta * lin        # (B,T,6)
    diff *= mask
    denom = tf.reduce_sum(mask) * tf.cast(tf.shape(real_sensors)[-1], tf.float32)
    return tf.reduce_sum(diff) / (denom + 1e-8)

class AdaptiveLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, target_ratio, adjustment_factor):
        self.initial_lr = initial_lr
        self.target_ratio = target_ratio  # Target ratio (e.g., D_loss should be ~0.8 * G_loss)
        self.adjustment_factor = adjustment_factor
    def __call__(self, d_loss, g_loss, name):
        ratio = d_loss / (g_loss + 1e-8)  # Avoid division by zero
        # Determine new learning rate based on ratio
        if ratio < self.target_ratio:
            new_lr = self.initial_lr * self.adjustment_factor
            print(f"Increasing {name} lr: {new_lr}")
        elif ratio > self.target_ratio:
            new_lr = self.initial_lr / self.adjustment_factor
            print(f"Decreasing {name} lr: {new_lr}")
        else:
            new_lr = self.initial_lr  # Keep LR constant
        
        # Update the initial_lr for the next epoch so that adjustments are cumulative
        self.initial_lr = new_lr  
        return new_lr

class BalancedAdaptiveLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 initial_gen_lr, 
                 initial_disc_lr, 
                 adjustment_factor=1.2, 
                 tolerance=0.05, 
                 min_lr=1e-6,
                 max_lr=1e-2):
        self.gen_lr = initial_gen_lr
        self.disc_lr = initial_disc_lr
        self.adjustment_factor = adjustment_factor
        self.tolerance = tolerance
        self.min_lr = min_lr
        self.max_lr = max_lr

    def __call__(self, d_loss, g_loss):
        ratio = d_loss / (g_loss + 1e-8)
        print(f"Current d_loss/g_loss ratio: {ratio:.2f}")

        # If within the tolerance, return current rates.
        if abs(ratio - 1.0) < self.tolerance:
            print(f"Learning rates remain unchanged. "
                  f"gen_lr: {self.gen_lr:.2e}, disc_lr: {self.disc_lr:.2e}")
            return self.gen_lr, self.disc_lr

        if ratio > 1:
            # Discriminator loss is higher: REDUCE disc LR to stabilize it
            new_disc_lr = self.disc_lr / self.adjustment_factor  # CHANGED: now reducing
            new_gen_lr = self.gen_lr * self.adjustment_factor   # Give generator more freedom
            print(f"Adjusting: Decreasing discriminator lr to {new_disc_lr:.2e} "
                  f"and increasing generator lr to {new_gen_lr:.2e}")
        else:
            # Generator loss is higher: REDUCE gen LR to stabilize it
            new_disc_lr = self.disc_lr * self.adjustment_factor  # Give discriminator more freedom
            new_gen_lr = self.gen_lr / self.adjustment_factor   # CHANGED: now reducing
            print(f"Adjusting: Increasing discriminator lr to {new_disc_lr:.2e} "
                  f"and decreasing generator lr to {new_gen_lr:.2e}")

        # Enforce min/max constraints
        self.gen_lr = max(min(new_gen_lr, self.max_lr), self.min_lr)
        self.disc_lr = max(min(new_disc_lr, self.max_lr), self.min_lr)

        return self.gen_lr, self.disc_lr

def analyze_spectral_characteristics(sensor_data, sampling_rate=30, 
                                   window_sizes=[32, 64, 128], 
                                   plot_results=False):
    """
    Analyze spectral characteristics of real swimming data across different window sizes
    and generate empirical weights for both window sizes and frequency bands.
    
    Args:
        csv_file: Path to CSV file containing swimming IMU data
        sampling_rate: Data sampling rate in Hz
        window_sizes: List of window sizes to analyze
        plot_results: Whether to plot the results
    
    Returns:
        Dictionary with spectral energy distribution, window weights, and frequency band weights
    """

    # Check if we have enough data
    if sensor_data.shape[0] < max(window_sizes):
        print(f"Warning: Data length ({len(sensor_data)}) is less than largest window size ({max(window_sizes)})")
        # Adjust window sizes if necessary
        window_sizes = [w for w in window_sizes if w <= len(sensor_data)]
        if not window_sizes:
            print("Error: No valid window sizes for this data length")
            return {}, {}, {}
    
  #  print(f"Analyzing {sensor_data.shape[0]} time points with {sensor_data.shape[1]} channels")
    
    # Define frequency bands for analysis
    # These should match the bands used in frequency_weighted_spectral_loss
    freq_bands = [
        (0, 1.0),     # 0-1 Hz (body position)
        (1.0, 3.0),   # 1-3 Hz (stroke frequency)
        (3.0, 5.0),   # 3-5 Hz (hand/foot movements)
        (5.0, 25.0)   # Higher frequencies (noise, impacts)
    ]
    band_names = ['0-1Hz', '1-3Hz', '3-5Hz', '5+Hz']
    
    # Define importance weights for each band based on swimming biomechanics
    # These are the default weights that will be adjusted based on data
    default_band_weights = {
        '0-1Hz': 0.47,    # Less important
        '1-3Hz': 2.0,    # Very important (stroke frequency)
        '3-5Hz': 0.9,    # Medium importance
        '5+Hz': 0.48      # Less important
    }
    
    results = {}
    
    # Prepare figure for plotting
    if plot_results:
        fig, axes = plt.subplots(len(window_sizes), 2, figsize=(15, 5*len(window_sizes)))
        if len(window_sizes) == 1:
            axes = np.array([axes])  # Ensure axes is always 2D
    
    # Initialize array to store frequency distribution across all window sizes
    all_freq_dist = np.zeros((len(window_sizes), sensor_data.shape[1], len(freq_bands)))
    
    for i, window_size in enumerate(window_sizes):
        # Initialize arrays to store spectral energy
        spectral_energy = np.zeros(sensor_data.shape[1])
        normalized_freq_dist = np.zeros((sensor_data.shape[1], len(freq_bands)))
        
        for channel_idx in range(sensor_data.shape[1]):
            # Get the entire time series for this channel
            signal_channel = sensor_data[:, channel_idx]
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(
                signal_channel, 
                fs=sampling_rate,
                nperseg=window_size,
                noverlap=window_size//2
            )
            
            # Calculate total energy
            total_energy = np.sum(Sxx)
            spectral_energy[channel_idx] = total_energy
            
            # Calculate energy in each band
            for band_idx, (low, high) in enumerate(freq_bands):
                band_indices = np.where((f >= low) & (f < high))[0]
                if len(band_indices) > 0:
                    band_energy = np.sum(Sxx[band_indices, :])
                    normalized_freq_dist[channel_idx, band_idx] = band_energy / total_energy if total_energy > 0 else 0
            
            # Store frequency distribution for later analysis
            all_freq_dist[i, channel_idx] = normalized_freq_dist[channel_idx]
              
        # Store results
        results[f'window_{window_size}'] = {
            'mean_energy': spectral_energy,
            'freq_distribution': normalized_freq_dist
        }
    
    # Calculate window weights
    # Determine which window size is best for each frequency band
    best_windows = {}
    for band_idx, band_name in enumerate(band_names):
        best_energy = -1
        best_window = None
        
        for window_idx, window_size in enumerate(window_sizes):
            # Average across all channels
            avg_energy = np.mean(all_freq_dist[window_idx, :, band_idx])
            
            if avg_energy > best_energy:
                best_energy = avg_energy
                best_window = window_size
        
        best_windows[band_name] = (best_window, best_energy)
    
    # Calculate importance of each window size
    window_importance = {}
    for window_idx, window_size in enumerate(window_sizes):
        # Calculate weighted importance based on which bands this window captures best
        importance = 0
        for band_idx, band_name in enumerate(band_names):
            # How well does this window capture this band?
            avg_energy = np.mean(all_freq_dist[window_idx, :, band_idx])
            
            # If this window is the best for this band, give it full weight
            if best_windows[band_name][0] == window_size:
                importance += default_band_weights[band_name] * 1.0
            else:
                # Otherwise, weight by relative energy compared to the best
                relative_energy = avg_energy / best_windows[band_name][1] if best_windows[band_name][1] > 0 else 0
                importance += default_band_weights[band_name] * relative_energy
        
        window_importance[window_size] = importance
    
    # Normalize importance to get weights that sum to 1.0
    total_window_importance = sum(window_importance.values())
    window_weights = {w: i/total_window_importance for w, i in window_importance.items()} if total_window_importance > 0 else {w: 1.0/len(window_sizes) for w in window_sizes}
    
    # Calculate frequency band weights based on energy distribution
    # Average across all channels and window sizes
    avg_freq_dist = np.mean(all_freq_dist, axis=(0, 1))  # Shape: [num_bands]
    
    # Adjust default weights based on observed energy distribution
    adjusted_band_weights = {}
    for band_idx, band_name in enumerate(band_names):
        # Scale the default weight by the observed energy
        # Higher energy in a band means it's more important to capture correctly
        energy_factor = avg_freq_dist[band_idx] * 2  # Scale factor
        
        # Ensure the factor is at least 0.5 to preserve some weight even for low-energy bands
        energy_factor = max(0.5, energy_factor)
        
        # Apply the energy factor to the default weight
        adjusted_band_weights[band_name] = default_band_weights[band_name] * energy_factor
    
    # Normalize to create weights that sum to a reasonable value
    total_band_weight = sum(adjusted_band_weights.values())
    normalized_band_weights = {k: v/total_band_weight * len(adjusted_band_weights) for k, v in adjusted_band_weights.items()}
    
    # Create a frequency weights array in the format needed for frequency_weighted_spectral_loss
    # For a typical FFT length
    fft_length = 1024  # Example FFT length
    freq_resolution = sampling_rate / fft_length
    num_freq_bins = fft_length // 2 + 1  # For RFFT
    
    # Calculate bin indices for specific frequency bands
    band_indices = [0]  # Start with 0
    for band_end, _ in freq_bands:
        band_indices.append(int(band_end / freq_resolution) + 1)
    
    # Ensure the last index doesn't exceed num_freq_bins
    band_indices[-1] = min(band_indices[-1], num_freq_bins)
    
    # Create frequency weights array
    freq_weights_array = []
    for i in range(len(band_names)):
        start_idx = band_indices[i]
        end_idx = band_indices[i+1] if i+1 < len(band_indices) else num_freq_bins
        weight = normalized_band_weights[band_names[i]]
        freq_weights_array.append(np.ones(end_idx - start_idx) * weight)
    
    # Combine into a single array
    freq_weights = np.concatenate(freq_weights_array, dtype=np.float32)
    
    # Ensure the array has the correct length
    if len(freq_weights) < num_freq_bins:
        # Pad with the last weight if needed
        freq_weights = np.pad(freq_weights, (0, num_freq_bins - len(freq_weights)), 
                             mode='constant', constant_values=freq_weights[-1])
    elif len(freq_weights) > num_freq_bins:
        # Truncate if too long
        freq_weights = freq_weights[:num_freq_bins]
    
    # Print recommendations
 #   print("\n=== SPECTRAL ANALYSIS RECOMMENDATIONS ===")
    
    # Print window size recommendations
  #  print("\nRecommended window size weights:")
  #  for window_size, weight in window_weights.items():
   #     window_type = "Short" if window_size == min(window_sizes) else \
   #                  "Long" if window_size == max(window_sizes) else "Medium"
    #    print(f"  {window_type} window ({window_size}): {weight:.2f}")
    
    # Print frequency band recommendations
  #  print("\nRecommended frequency band weights:")
  #  for band_name, weight in normalized_band_weights.items():
  #      print(f"  {band_name}: {weight:.2f}")
    
    # Format for direct use in code
  #  window_weights_str = ", ".join([f"{window_weights[w]:.2f}" for w in window_sizes])
 #   print(f"\nFor comprehensive_spectral_loss: [{window_weights_str}]")
    
  #  freq_weights_str = ", ".join([f"{normalized_band_weights[b]:.2f}" for b in band_names])
  #  print(f"\nFor frequency_weighted_spectral_loss: [{freq_weights_str}]")
    
    return results, window_weights, {
        'normalized_weights': normalized_band_weights,
        'freq_weights_array': freq_weights,
        'band_indices': band_indices
    }


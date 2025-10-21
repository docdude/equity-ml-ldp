import numpy as np
import umap
from scipy.spatial.distance import cosine, jensenshannon
from scipy.spatial.distance import euclidean as scipy_euclidean
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import pairwise_kernels

from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import tensorflow as tf

def compute_stroke_similarity(real_strokes, fake_strokes):
    """
    Computes the percentage similarity between real and generated stroke counts.
    """
    #real_strokes = real_samples[..., 12]
    #fake_strokes = fake_samples[..., 12].numpy()

    # Round the stroke outputs at a threshold, e.g. 0.5
    fake_strokes = np.where(fake_strokes > 0.5, 1, 0)
    real_stroke_count = np.sum(real_strokes)  # Total real stroke occurrences
    fake_stroke_count = np.sum(fake_strokes)  # Total fake stroke occurrences

    similarity = 100 - (np.abs(real_stroke_count - fake_stroke_count) / (real_stroke_count + 1e-8)) * 100
    return max(similarity, 0)  # Ensure it's non-negative


def compute_swim_style_similarity_js(real_styles, fake_styles):
    """
    Computes Jensen-Shannon similarity of real vs. fake swim style distributions per timestep.
    
    Args:
        real_styles (numpy array): Shape (num_samples, 180, 6) containing real label data.
        fake_styles (numpy array): Shape (num_samples, 180, 6) containing fake label data.
    
    Returns:
        float: Percentage similarity between real and fake swim style distributions.
    """

    # Extract swim styles (one-hot encoded in columns 6-12)
    real_styles = np.argmax(real_styles, axis=-1)  # Shape: (num_samples, 180)
    fake_styles = np.argmax(fake_styles, axis=-1)  # Shape: (num_samples, 180)

    #num_samples = real_styles.shape[0]  # Number of samples
    time_steps = real_styles.shape[1]   # 180 time steps

    js_divs = []

    for t in range(time_steps):  # Iterate over each time step
        # Count occurrences of each swim style at timestep t across all samples
        real_counts = np.bincount(real_styles[:, t], minlength=6)  # Count per class
        fake_counts = np.bincount(fake_styles[:, t], minlength=6)

        # Convert counts to probability distributions
        real_dist = real_counts / (np.sum(real_counts) + 1e-8)
        fake_dist = fake_counts / (np.sum(fake_counts) + 1e-8)

        # Compute JS divergence (stable alternative to KL divergence)
        js_div = jensenshannon(real_dist, fake_dist, base=2)
        js_divs.append(js_div)

    # Average JSD over all timesteps
    avg_js_div = np.mean(js_divs)

    # Convert JSD to a similarity percentage
    similarity = 100 - avg_js_div * 100  # Lower JSD means higher similarity
    return max(similarity, 0)  # Ensure it's non-negative


def compute_swim_style_similarity_match(real_styles, fake_styles):
    """
    Computes per-instance, per-timestep similarity of real vs. fake swim style distributions.

    Args:
        real_samples (numpy array): Shape (num_samples, 180, 13) containing real sensor & label data.
        fake_samples (numpy array): Shape (num_samples, 180, 13) containing fake sensor & label data.

    Returns:
        float: Percentage similarity between real and fake swim style distributions over all instances.
    """

    # Extract swim styles (one-hot encoded in columns 6-12)
    #real_styles = np.argmax(real_styles, axis=-1).flatten()  # Shape: (num_samples * 180,)
    real_styles = real_styles.flatten()
    fake_styles = np.round(fake_styles.flatten())#np.argmax(fake_styles, axis=-1).flatten()  # Shape: (num_samples * 180,)

    # Compute match percentage across all instances (vectorized instead of looping)
    similarity_percentage = np.mean(real_styles == fake_styles) * 100

    return similarity_percentage

def compute_swim_style_similarity_match_mse(real_styles, fake_styles):
    """
    Computes per-instance, per-timestep similarity of real vs. fake swim style distributions.

    Args:
        real_samples (numpy array): Shape (num_samples, 180, 13) containing real sensor & label data.
        fake_samples (numpy array): Shape (num_samples, 180, 13) containing fake sensor & label data.

    Returns:
        float: Percentage similarity between real and fake swim style distributions over all instances.
    """

    real_styles = real_styles.flatten()
    fake_styles = np.round(fake_styles.flatten())
    # Compute match percentage across all instances (vectorized instead of looping)
    similarity_percentage = np.mean(real_styles == fake_styles) * 100

    return similarity_percentage


from sklearn.metrics.pairwise import cosine_similarity

def compute_swim_style_similarity_match_cos(real_styles, fake_styles):
    """
    Computes per-instance, per-timestep similarity of real vs. fake swim style distributions 
    using cosine similarity to handle soft one-hot labels properly.

    Args:
        real_styles (numpy array): Shape (num_samples, 180, num_styles) - Real soft one-hot labels.
        fake_styles (numpy array): Shape (num_samples, 180, num_styles) - Fake soft one-hot labels.

    Returns:
        float: Average cosine similarity between real and fake swim style distributions over all instances.
    """

    num_samples, time_steps, num_styles = real_styles.shape  # Ensure same shape for real and fake

    # Flatten along batch & time steps while keeping style dimension
    real_flat = real_styles.reshape(-1, num_styles)  # Shape: (num_samples * 180, num_styles)
    fake_flat = fake_styles.reshape(-1, num_styles)  # Shape: (num_samples * 180, num_styles)

    # Compute cosine similarity for each (real, fake) distribution pair
    similarities = np.array([cosine_similarity(r.reshape(1, -1), f.reshape(1, -1))[0, 0]
                             for r, f in zip(real_flat, fake_flat)])

    # Compute the mean similarity score across all instances
    avg_similarity = np.mean(similarities) * 100  # Convert to percentage

    return avg_similarity

def compute_sensor_similarity_per_channel(real_sensors, fake_sensors):
    """
    Computes similarity based on mean, std deviation, and correlation.
    """
    #real_sensors = real_samples[..., :6]
    #fake_sensors = fake_samples[..., :6].numpy()
    # Compute mean & std per channel
    real_mean = np.mean(real_sensors, axis=(0, 1))  # Shape: (6,)
    real_std = np.std(real_sensors, axis=(0, 1))
    fake_mean = np.mean(fake_sensors, axis=(0, 1))
    fake_std = np.std(fake_sensors, axis=(0, 1))

    # Mean & Std Differences (Avoid division by near-zero)
    mean_diff = np.mean(np.abs(real_mean - fake_mean) / np.maximum(np.abs(real_mean), 1e-8))
    std_diff = np.mean(np.abs(real_std - fake_std) / np.maximum(np.abs(real_std), 1e-8))

    # Compute Pearson correlation per sensor channel separately
    correlation_per_channel = []
    for c in range(6):  # Iterate over 6 sensor channels
        real_flat = real_sensors[:, :, c].flatten()  # Flatten across all samples & timesteps
        fake_flat = fake_sensors[:, :, c].flatten()

        if np.std(real_flat) == 0 or np.std(fake_flat) == 0:
            corr = 0  # Avoid division by zero in np.corrcoef
        else:
            corr = np.corrcoef(real_flat, fake_flat)[0, 1]

        correlation_per_channel.append(corr)

    correlation = np.mean(correlation_per_channel)  # Average correlation over all 6 channels

    # Compute overall similarity score
    similarity = 100 - (mean_diff + std_diff + (1 - correlation)) * 100
    print(f"Sensor Mean Diff: {mean_diff:.2f} | Sensor Std Diff: {std_diff:.2f} | Sensor Correlation: {correlation:.2f}")
    return max(similarity, 0)  # Ensure non-negative similarity


def compute_sensor_similarity(real_sensors, fake_sensors):
    """
    Computes similarity based on mean, std deviation, and combined correlation across all channels.
    """
    # Compute mean & std for all channels combined
    real_mean = np.mean(real_sensors)  # Mean across all samples, timesteps, and channels
    real_std = np.std(real_sensors)    # Std across all samples, timesteps, and channels
    fake_mean = np.mean(fake_sensors)  # Mean across all samples, timesteps, and channels
    fake_std = np.std(fake_sensors)    # Std across all samples, timesteps, and channels
    real_min = np.min(real_sensors)
    real_max = np.max(real_sensors)
    fake_min = np.min(fake_sensors)
    fake_max = np.max(fake_sensors)
    # Compute mean & std differences (Avoid division by near-zero)
    mean_diff = np.abs(real_mean - fake_mean) / (np.abs(real_mean) + np.abs(fake_mean) + 1e-8)
    std_diff = np.abs(real_std - fake_std) / (np.abs(real_std) + 1e-8)

    # Flatten all data across samples, timesteps, and channels for combined correlation
    real_flat = real_sensors.flatten()
    fake_flat = fake_sensors.flatten()

    # Compute Pearson correlation across all channels combined
    if np.std(real_flat) == 0 or np.std(fake_flat) == 0:
        correlation = 0  # Avoid division by zero in np.corrcoef
    else:
        correlation = np.corrcoef(real_flat, fake_flat)[0, 1]

    # Compute overall similarity score
    # Adjust weights for mean_diff, std_diff, and correlation as needed
    #similarity = 100 - (mean_diff * 50 + std_diff * 30 + (1 - correlation) * 20)
    similarity = 100 - (mean_diff + std_diff + (1 - correlation)) * 100

    similarity = max(similarity, 0)  # Ensure non-negative 

    print(f"   Real sensor stats | Mean: {real_mean:.2f}, Std: {real_std:.2f}, Min: {real_min:.2f}, Max: {real_max:.2f}")
    print(f"   Fake sensor stats | Mean: {fake_mean:.2f}, Std: {fake_std:.2f}, Min: {fake_min:.2f}, Max: {fake_max:.2f}")
    if abs(real_mean - fake_mean) > 0.5:
        print(f"Significant mean discrepancy: Real {real_mean:.2f} vs Fake {fake_mean:.2f}")
 
    print(f"Sensor Mean Diff: {mean_diff:.2f} | Sensor Std Diff: {std_diff:.2f} | Sensor Correlation: {correlation:.2f}")


    return similarity


def compute_js_divergence_3d_dynamic_bins(real_sensors, fake_sensors):
    """
    Computes the Jensen-Shannon Divergence (JSD) between real and fake sensor data
    for 3D data (samples, timesteps, channels) with dynamic binning.
    
    Parameters:
        real_sensors (numpy.ndarray): Real sensor data of shape (samples, timesteps, channels).
        fake_sensors (numpy.ndarray): Fake sensor data of shape (samples, timesteps, channels).
    
    Returns:
        float: Average Jensen-Shannon Divergence across all sensor channels.
    """
    # Ensure the input is 3D (samples, timesteps, channels)
    if real_sensors.ndim != 3 or fake_sensors.ndim != 3:
        raise ValueError("Input sensor data must be 3D arrays (samples, timesteps, channels).")
    
    # Number of sensor channels
    num_channels = real_sensors.shape[2]
    
    # Initialize list to store JSD for each channel
    js_divergences = []
    
    for channel in range(num_channels):
        # Extract data for the current channel across all samples and timesteps
        real_channel = real_sensors[:, :, channel].flatten()  # Flatten to 1D
        fake_channel = fake_sensors[:, :, channel].flatten()  # Flatten to 1D
        
        # Combine real and fake data to determine dynamic bin edges
        combined_data = np.concatenate([real_channel, fake_channel])
        
        # Compute bin width using Freedman-Diaconis rule
        q75, q25 = np.percentile(combined_data, [75, 25])  # Compute IQR
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(combined_data) ** (1 / 3))  # Freedman-Diaconis rule
        
        # If IQR is zero or bin_width is too small, fall back to a default bin width
        if bin_width <= 0:
            bin_width = (np.max(combined_data) - np.min(combined_data)) / 50  # Default to 50 bins
        
        # Compute the number of bins
        num_bins = int(np.ceil((np.max(combined_data) - np.min(combined_data)) / bin_width))
        
        # Compute histograms (probability distributions) for real and fake data
        real_hist, bin_edges = np.histogram(real_channel, bins=num_bins, density=True)
        fake_hist, _ = np.histogram(fake_channel, bins=bin_edges, density=True)
        
        # Add a small epsilon to avoid division by zero or log(0)
        epsilon = 1e-8
        real_hist = real_hist + epsilon
        fake_hist = fake_hist + epsilon
        
        # Normalize histograms to ensure they sum to 1 (probability distributions)
        real_hist /= np.sum(real_hist)
        fake_hist /= np.sum(fake_hist)
        
        # Compute Jensen-Shannon Divergence using scipy's jensenshannon function
        js_div = jensenshannon(real_hist, fake_hist, base=2)  # Base 2 for information entropy
        js_divergences.append(js_div)
    
    # Average JSD across all channels
    avg_js_divergence = np.mean(js_divergences)
    
    # Convert JSD to a similarity percentage
    similarity = 100 - avg_js_divergence * 100  # Lower JSD means higher similarity
    return max(similarity, 0)  # Ensure it's non-negative


def parse_samples(real_samples, fake_samples):
    real_strokes = real_samples[..., 12]
    fake_strokes = fake_samples[..., 12].numpy()

    real_styles = real_samples[..., 6:12]
    fake_styles = fake_samples[..., 6:12].numpy()
   # print("Sum of each row (should be ~1):", np.sum(real_styles, axis=-1))
    # Ensure numerical stability in softmax sum
    #fake_styles /= np.clip(np.sum(fake_styles, axis=-1, keepdims=True), 1e-8, 1.0)
    
    #print("Sum of each row (should be ~1):", np.sum(fake_styles, axis=-1))

    real_sensors = real_samples[..., :6]
    fake_sensors = fake_samples[..., :6].numpy()

    return real_strokes, fake_strokes, real_styles, fake_styles, real_sensors, fake_sensors

def parse_samples_sce(real_samples, fake_samples):
    real_strokes = real_samples[..., 7]
    fake_strokes = fake_samples[..., 12].numpy()

    real_styles = real_samples[..., 6]
    fake_styles = fake_samples[..., 6:12].numpy()
   # print("Sum of each row (should be ~1):", np.sum(real_styles, axis=-1))
    # Ensure numerical stability in softmax sum
    #fake_styles /= np.clip(np.sum(fake_styles, axis=-1, keepdims=True), 1e-8, 1.0)
    
    #print("Sum of each row (should be ~1):", np.sum(fake_styles, axis=-1))

    real_sensors = real_samples[..., :6]
    fake_sensors = fake_samples[..., :6].numpy()

    return real_strokes, fake_strokes, real_styles, fake_styles, real_sensors, fake_sensors

def parse_samples_mse(real_samples, fake_samples):
    real_strokes = real_samples[..., 7]
    fake_strokes = fake_samples[..., 7].numpy()

    real_styles = real_samples[..., 6]
    fake_styles = fake_samples[..., 6].numpy()
   # print("Sum of each row (should be ~1):", np.sum(real_styles, axis=-1))
    # Ensure numerical stability in softmax sum
    #fake_styles /= np.clip(np.sum(fake_styles, axis=-1, keepdims=True), 1e-8, 1.0)
    
    #print("Sum of each row (should be ~1):", np.sum(fake_styles, axis=-1))

    real_sensors = real_samples[..., :6]
    fake_sensors = fake_samples[..., :6].numpy()

    return real_strokes, fake_strokes, real_styles, fake_styles, real_sensors, fake_sensors

def compute_umap_similarity(real_sensors, fake_sensors, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Uses UMAP to project real and fake sensor data into a lower-dimensional space
    and computes similarity based on distribution overlap.
    """
    #real_sensors = real_samples[..., :6].reshape(real_samples.shape[0], -1)  # Flatten over time
    #fake_sensors = fake_samples[..., :6].reshape(fake_samples.shape[0], -1)
    real_sensors = real_sensors.reshape(real_sensors.shape[0], -1)
    fake_sensors = fake_sensors.reshape(fake_sensors.shape[0], -1)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    real_embedded = reducer.fit_transform(real_sensors)
    fake_embedded = reducer.transform(fake_sensors)
    
    umap_distance = np.mean(np.linalg.norm(real_embedded - fake_embedded, axis=1))
    similarity = 100 - (umap_distance * 10)  # Scale to percentage
    return max(similarity, 0)

def compute_rmse(real_sensors, fake_sensors):
    """
    Computes RMSE (Root Mean Square Error) between real and fake sensor data.
    Normalizes per-channel RMSE to account for scale differences.
    """
    real_std = np.std(real_sensors, axis=(0, 1)) + 1e-8  # Prevent division by zero
    rmse_per_channel = [np.sqrt(mean_squared_error(real_sensors[..., i], fake_sensors[..., i])) / real_std[i] for i in range(6)]
    avg_rmse = np.mean(rmse_per_channel)
    return avg_rmse

from sklearn.metrics import mean_squared_error
import numpy as np

def compute_rmse_per_swim_style(real_sensors, fake_sensors, real_styles):
    """
    Computes RMSE per swim style, averaging over all six sensor channels.

    Args:
        real_sensors (numpy array): Real sensor data, shape (num_samples, 180, 6).
        fake_sensors (numpy array): Fake sensor data, shape (num_samples, 180, 6).
        real_styles (numpy array): One-hot encoded swim styles, shape (num_samples, 180, 6).

    Returns:
        float: Average RMSE across all swim styles.
    """
    num_styles = real_styles.shape[-1]  # Assuming one-hot encoding in last dimension
    rmse_per_style = []

    for style_idx in range(num_styles):
        # Mask samples belonging to this swim style
        style_mask = real_styles[..., style_idx] == 1  # Shape: (num_samples, 180)
        
        if not np.any(style_mask):
            continue  # Skip styles with no data
        
        # Extract only the selected swim style's sensor data
        real_selected = real_sensors[style_mask]  # Shape: (selected_samples, 6)
        fake_selected = fake_sensors[style_mask]  # Shape: (selected_samples, 6)

        if real_selected.shape[0] == 0 or fake_selected.shape[0] == 0:
            continue  # Skip if no matching samples exist

        # Compute RMSE per channel
        rmse_per_channel = [
            np.sqrt(mean_squared_error(real_selected[:, i], fake_selected[:, i])) 
            for i in range(6)
        ]
        
        rmse_per_style.append(np.mean(rmse_per_channel))

    # Average RMSE across all swim styles
    return np.mean(rmse_per_style) if rmse_per_style else 0.0  # Avoid NaN if no styles exist


def compute_prd_rmse(real_sensors, fake_sensors):
    """
    Computes the Percent Root-Mean-Square Difference (PRD) between real and fake sensor data.

    Args:
        real_sensors (numpy array): Real sensor data, shape (num_samples, timesteps, 6).
        fake_sensors (numpy array): Fake sensor data, shape (num_samples, timesteps, 6).

    Returns:
        float: PRD score in percentage, lower values indicate higher similarity.
    """
    # Ensure the inputs have the same shape
    if real_sensors.shape != fake_sensors.shape:
        raise ValueError("Shape mismatch: Real and Fake sensors must have the same shape.")

    # Compute Root-Mean-Square (RMS) of the difference
    rms_diff = np.sqrt(np.mean((real_sensors - fake_sensors) ** 2, axis=(0, 1)))  # Per channel

    # Compute RMS of real signals
    rms_real = np.sqrt(np.mean(real_sensors ** 2, axis=(0, 1)))  # Per channel

    # Compute PRD percentage
    prd_per_channel = 100 * (rms_diff / (rms_real + 1e-8))  # Prevent division by zero

    # Average PRD over all 6 sensor channels
    prd_score = np.mean(prd_per_channel)

    return prd_score

def custom_euclidean(u, v):
    """
    Custom Euclidean distance function that handles both scalars and arrays.
    
    Parameters:
    -----------
    u : float or ndarray
        First input value or array.
    v : float or ndarray
        Second input value or array.
    
    Returns:
    --------
    float
        Computed Euclidean distance.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Check if inputs are scalars
    if u.ndim == 0 and v.ndim == 0:
        return abs(u - v)  # Return absolute difference for scalars
    else:
        return scipy_euclidean(u, v)  # Use scipy's Euclidean for arrays

def compute_dtw_distance(real_sensors, fake_sensors):
    """
    Computes the DTW (Dynamic Time Warping) distance between real and fake sensor sequences
    for each channel and averages the distances across all channels.
    
    Parameters:
        real_sensors (numpy.ndarray): Real sensor data of shape (samples, timesteps, channels).
        fake_sensors (numpy.ndarray): Fake sensor data of shape (samples, timesteps, channels).
    
    Returns:
        float: Average DTW distance across all channels.
    """
    # Ensure the input is 3D (samples, timesteps, channels)
    if real_sensors.ndim != 3 or fake_sensors.ndim != 3:
        raise ValueError("Input sensor data must be 3D arrays (samples, timesteps, channels).")
    
    # Check that real and fake data have the same shape
    if real_sensors.shape != fake_sensors.shape:
        raise ValueError("Real and fake sensor data must have the same shape.")
    
    num_samples, num_timesteps, num_channels = real_sensors.shape
    
    dtw_distances = []
    
    # Iterate over each channel
    for channel in range(num_channels):
        channel_distances = []
        
        # Compute DTW distance for each sample in the current channel
        for sample in range(num_samples):
            real_sequence = real_sensors[sample, :, channel]  # 1D time-series for real data
            fake_sequence = fake_sensors[sample, :, channel]  # 1D time-series for fake data
            #print(f"Processing sample {sample}, channel {channel}: real_sequence shape {real_sequence.shape}, fake_sequence shape {fake_sequence.shape}")
            real_sequence = (real_sequence - np.mean(real_sequence)) / (np.std(real_sequence) + 1e-8)
            fake_sequence = (fake_sequence - np.mean(fake_sequence)) / (np.std(fake_sequence) + 1e-8)

            # Compute DTW distance between the two sequences
            distance, _ = fastdtw(real_sequence, fake_sequence, dist=custom_euclidean)#dist=lambda x, y: abs(x - y))
            channel_distances.append(distance)
        
        # Average DTW distance for the current channel
        dtw_distances.append(np.mean(channel_distances))
    
    # Average DTW distance across all channels
    avg_dtw = np.mean(dtw_distances)
    return avg_dtw

def compute_cosine_similarity(real_sensors, fake_sensors):
    """
    Computes the cosine similarity between real and fake sensor data.
    """
    #real_sensors = real_samples[..., :6].reshape(real_samples.shape[0], -1)  # Flatten over time
    #fake_sensors = fake_samples[..., :6].reshape(fake_samples.shape[0], -1)
    real_sensors = real_sensors.reshape(real_sensors.shape[0], -1)
    fake_sensors = fake_sensors.reshape(fake_sensors.shape[0], -1)  
    
    cosine_similarities = [1 - cosine(real_sensors[i], fake_sensors[i]) for i in range(real_sensors.shape[0])]
    avg_cosine_similarity = np.mean(cosine_similarities) * 100  # Scale to percentage
    return max(avg_cosine_similarity, 0)

def compute_sensor_similarity_combined(real_sensors, fake_sensors):
    """
    Computes an overall sensor similarity score using multiple metrics.
    """
    umap_sim = compute_umap_similarity(real_sensors, fake_sensors)
    rmse_score = compute_rmse(real_sensors, fake_sensors)
    dtw_score = compute_dtw_distance(real_sensors, fake_sensors)
    cosine_sim = compute_cosine_similarity(real_sensors, fake_sensors)
    
    # Combine scores (lower RMSE/DTW means better, so we invert them)
    overall_similarity = (0.4 * umap_sim) + (0.2 * (100 - rmse_score * 10)) + \
                         (0.2 * (100 - dtw_score * 0.1)) + (0.2 * cosine_sim)
    
    return max(overall_similarity, 0), umap_sim, rmse_score, dtw_score, cosine_sim


def compute_frechet_distance_old(real_samples, fake_samples):
    """
    Computes the Fréchet Distance (FD) between the real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
    
    Returns:
        float: Fréchet Distance score (lower is better)
    """
    real_samples = real_samples.reshape(-1, real_samples.shape[-1])  # Flatten time dimension
    fake_samples = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    # Compute mean and covariance of both distributions
    mu_real, sigma_real = np.mean(real_samples, axis=0), np.cov(real_samples, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_samples, axis=0), np.cov(fake_samples, rowvar=False)
    
    # Compute the squared difference of means
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Compute sqrt of product of covariances
    try:
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
    except Exception as e:
        print("Covariance sqrt error:", e)
        cov_sqrt = np.zeros_like(sigma_real)
    
    # Compute Fréchet Distance
    fd = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return max(fd, 0)  # Ensure non-negative value


def compute_frechet_distance(real_samples, fake_samples, eps=1e-6, per_channel=False, min_samples=1000):
    """
    Computes the Fréchet Distance (FD) between the real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        eps (float): Small constant for numerical stability
        per_channel (bool): If True, compute FD for each channel separately
        min_samples (int): Minimum recommended samples for stable estimation
    
    Returns:
        float or dict: Overall FD or dict with overall and per-channel FDs
    """
    # Check input shapes
    assert real_samples.shape[2] == fake_samples.shape[2], "Channel dimensions must match"
    
    if per_channel:
        channel_fds = []
        channel_names = [f"Channel_{i}" for i in range(real_samples.shape[2])]
        
        for ch in range(real_samples.shape[2]):
            real_ch = real_samples[:, :, ch].flatten()  # One channel, flattened time
            fake_ch = fake_samples[:, :, ch].flatten()
            
            # Compute mean and variance (1D case)
            mu_real, sigma_real = np.mean(real_ch), np.var(real_ch)
            mu_fake, sigma_fake = np.mean(fake_ch), np.var(fake_ch)
            
            # Add epsilon for stability
            sigma_real += eps
            sigma_fake += eps
            
            # 1D Fréchet distance
            fd_ch = (mu_real - mu_fake)**2 + sigma_real + sigma_fake - 2*np.sqrt(sigma_real * sigma_fake)
            channel_fds.append(max(fd_ch, 0))
        
        # Also compute the joint FD
        joint_fd = compute_joint_fd(real_samples, fake_samples, eps, min_samples)
        
        # Return both overall and per-channel results
        return {
            "overall_fd": joint_fd,
            "channel_fd": dict(zip(channel_names, channel_fds)),
            "average_channel_fd": np.mean(channel_fds)
        }
    else:
        return compute_joint_fd(real_samples, fake_samples, eps, min_samples)

def compute_joint_fd(real_samples, fake_samples, eps=1e-6, min_samples=1000):
    """Helper function to compute joint FD across all channels"""
    real_flat = real_samples.reshape(-1, real_samples.shape[-1])
    fake_flat = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    # Check sample size
    if len(real_flat) < min_samples or len(fake_flat) < min_samples:
        print(f"Warning: Sample size may be too small for stable FD estimation. "
              f"Real: {len(real_flat)}, Fake: {len(fake_flat)}")
    
    # Compute mean and covariance
    mu_real, sigma_real = np.mean(real_flat, axis=0), np.cov(real_flat, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_flat, axis=0), np.cov(fake_flat, rowvar=False)
    
    # Add small epsilon to diagonal for numerical stability
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps
    
    # Compute squared difference of means
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Compute sqrt of product of covariances
    try:
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
    except Exception as e:
        print(f"Covariance sqrt error: {e}")
        # Return a large value instead of zeros
        return float('inf')
    
    # Compute Fréchet Distance
    fd = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return max(fd, 0)  # Ensure non-negative value

def compute_mmd(real_samples, fake_samples, kernel='rbf', sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) between real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        kernel (str): Type of kernel to use ('rbf' or 'linear')
        sigma (float): Kernel bandwidth for RBF kernel
    
    Returns:
        float: MMD score (lower is better)
    """
    real_samples = real_samples.reshape(-1, real_samples.shape[-1])  # Flatten time dimension
    fake_samples = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    if kernel == 'rbf':
        gamma = 1.0 / (2 * sigma ** 2)
        K_real = pairwise_kernels(real_samples, real_samples, metric='rbf', gamma=gamma)
        K_fake = pairwise_kernels(fake_samples, fake_samples, metric='rbf', gamma=gamma)
        K_cross = pairwise_kernels(real_samples, fake_samples, metric='rbf', gamma=gamma)
    else:
        K_real = pairwise_kernels(real_samples, real_samples, metric='linear')
        K_fake = pairwise_kernels(fake_samples, fake_samples, metric='linear')
        K_cross = pairwise_kernels(real_samples, fake_samples, metric='linear')
    
    mmd = np.mean(K_real) + np.mean(K_fake) - 2 * np.mean(K_cross)
    return max(mmd, 0)  # Ensure non-negative value

                
def plot_to_image(fig):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(fig)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def plot_samples(real_samples, fake_samples, start_idx=50, num_samples=5):
    """Plot all 6 sensor channels with stroke peak indicators"""
    plt.figure(figsize=(20, 10), facecolor='white')
    real_samples = real_samples[start_idx : start_idx + num_samples]   
    fake_samples = fake_samples[start_idx : start_idx + num_samples].numpy()     
   
    for i in range(num_samples):

        plt.subplot(2, num_samples, i+1)
        
        # Plot sensor channels 0-5 with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        for ch in range(3):  # Now plotting all 6 sensor channels
            sensor_data = real_samples[i, :, ch]
            plt.plot(sensor_data, 
                    color=colors[ch], 
                    alpha=0.7,
                    label=f'Sensor {ch+1}')
            
            # Add stroke peak indicators as scatter points (channel 7 is stroke label)
            if ch == 0:  # Only add to first sensor to avoid duplicate points
                stroke_indices = np.where(real_samples[i, :, 12] > 0.5)[0]
                plt.scatter(stroke_indices, sensor_data[stroke_indices],
                          color='red', marker='o', s=10,
                          edgecolor='black', zorder=5,
                          label='Stroke Peaks (Real)')
            
        plt.title(f'Real Sample {i+1}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Fake Sample
        plt.subplot(2, num_samples, num_samples+i+1)
        for ch in range(3):  # Now plotting all 6 sensor channels
            sensor_data = fake_samples[i, :, ch]
            plt.plot(sensor_data,
                    color=colors[ch],
                    alpha=0.7)
            
            # Add generated stroke indicators
            if ch == 0:  # Only add to first sensor
                fake_strokes = np.where(fake_samples[i, :, 12] > 0.5)[0]
                plt.scatter(fake_strokes, sensor_data[fake_strokes],
                          color='blue', marker='o', s=10,
                          edgecolor='black', zorder=5,
                          label='Stroke Peaks (Generated)')
            
        plt.title(f'Generated Sample {i+1}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()


def plot_swim_styles(real_styles, fake_styles):
    """
    Plots real vs. fake swim styles as two separate continuous graphs stacked vertically.

    Args:
        real_samples (numpy array): Real swim style predictions (batch_size, 180, 13).
        fake_samples (numpy array): Generated swim style predictions (batch_size, 180, 13).
    """
    num_samples = real_styles.shape[0]  # Number of samples
    time_steps = real_styles.shape[1]   # 180 time steps
    total_steps = num_samples * time_steps  # Total points to plot

    # Extract swim style predictions and convert to categorical (argmax across 6 style probabilities)
    real_styles = np.argmax(real_styles, axis=-1).flatten()  # Shape: (num_samples * 180,)
    fake_styles = np.argmax(fake_styles, axis=-1).flatten()  # Shape: (num_samples * 180,)

    # Scale x-axis to make visualization manageable
    x_values = np.arange(total_steps) # Normalize index for better visualization

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    # Real Swim Styles (Top Plot)
    axes[0].plot(x_values, real_styles, linestyle='-', color='blue', alpha=0.7)
    axes[0].set_ylabel("Real Swim Style")
    axes[0].set_yticks(range(6))  # Swim styles (0-5)
    axes[0].set_title("Real Swim Styles")

    # Fake Swim Styles (Bottom Plot)
    axes[1].plot(x_values, fake_styles, linestyle='-', color='red', alpha=0.7)
    axes[1].set_ylabel("Fake Swim Style")
    axes[1].set_yticks(range(6))  # Swim styles (0-5)
    axes[1].set_xlabel("Sample Index (scaled)")
    axes[1].set_title("Fake Swim Styles")

    plt.tight_layout()
    return plt.gcf()

def plot_swim_styles_soft(real_styles, fake_styles):
    """
    Plots real vs. fake swim styles as two separate continuous line graphs.
    
    Instead of using `argmax`, this function samples swim styles from the soft one-hot distributions
    using `np.random.choice`, which respects the probabilistic nature of the labels.

    Args:
        real_styles (numpy array): Real swim style distributions (batch_size, 180, num_styles).
        fake_styles (numpy array): Generated swim style distributions (batch_size, 180, num_styles).
    """
    num_samples, time_steps, num_styles = real_styles.shape  # Shape: (batch_size, 180, num_styles)

    # Convert soft one-hot labels into discrete swim styles by sampling
    row_real_styles = np.array([np.random.choice(num_styles, p=soft_row) for soft_row in real_styles.reshape(-1, num_styles)])
    row_fake_styles = np.array([np.random.choice(num_styles, p=soft_row) for soft_row in fake_styles.reshape(-1, num_styles)])

    # X-axis values (scaled by total samples)
    x_values = np.arange(len(row_real_styles))

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    # Real Swim Styles (Top Plot)
    axes[0].plot(x_values, row_real_styles, linestyle='-', color='blue', alpha=0.7)
    axes[0].set_ylabel("Real Swim Style")
    axes[0].set_yticks(range(num_styles))  # Swim styles (0-5)
    axes[0].set_title("Real Swim Styles")

    # Fake Swim Styles (Bottom Plot)
    axes[1].plot(x_values, row_fake_styles, linestyle='-', color='red', alpha=0.7)
    axes[1].set_ylabel("Fake Swim Style")
    axes[1].set_yticks(range(num_styles))  # Swim styles (0-5)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_title("Fake Swim Styles")

    plt.tight_layout()
    plt.show()
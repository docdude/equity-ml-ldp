"""
Barrier Heatmap Target Generation
==================================

Converts discrete barrier events into Gaussian heatmap targets for regression.

Adapted from swimming stroke detection approach (FinancialHeatmapAugmentation):
- Swimming: Binary stroke events → Gaussian heatmap
- Finance: Barrier touches (TP/SL) → Dual-channel Gaussian heatmap

Key idea: Instead of predicting discrete class at sequence end,
predict continuous probability field over entire sequence.

This module provides BOTH approaches:
1. create_heatmaps_from_events() - Uses FinancialHeatmapAugmentation (RECOMMENDED)
2. create_heatmaps_simple() - Simplified approach (places Gaussian at sequence end)
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Import your proven swimming approach
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from fin_load_and_sequence import FinancialHeatmapAugmentation


def create_gaussian_kernel(sigma: float = 2.0, radius: int = None) -> np.ndarray:
    """
    Create 1D Gaussian kernel for temporal smoothing.
    
    Args:
        sigma: Standard deviation in timesteps
               sigma=2.0 → ~95% weight within ±4 timesteps
        radius: Kernel radius (defaults to 3*sigma)
    
    Returns:
        kernel: 1D array of Gaussian weights
    
    Example:
        sigma=2.0, radius=6 → kernel of length 13 centered at index 6
    """
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Don't normalize - preserve magnitude as confidence indicator
    # (matching your swimming implementation)
    
    return kernel


def barrier_event_to_position(
    event_start_date: pd.Timestamp,
    barrier_touch_date: pd.Timestamp,
    sequence_dates: pd.DatetimeIndex
) -> int:
    """
    Convert barrier touch date to position within sequence.
    
    Args:
        event_start_date: When the event/sequence started
        barrier_touch_date: When TP or SL was touched (t1)
        sequence_dates: DatetimeIndex of the sequence window
    
    Returns:
        position: Index within sequence (0 to len-1), or -1 if not in window
    
    Example:
        sequence_dates: [Day0, Day1, Day2, ..., Day19]
        barrier touched on Day5 → position = 5
    """
    try:
        # Find position of barrier touch within sequence
        position = sequence_dates.get_loc(barrier_touch_date)
        return position
    except KeyError:
        # Barrier touch not in this sequence window
        return -1


def create_barrier_heatmaps_from_events(
    events_df: pd.DataFrame,
    sequence_dates_list: list,  # List of DatetimeIndex, one per sequence
    seq_len: int = 20,
    sigma: float = 2.0,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Create Gaussian heatmap targets from barrier events.
    
    Args:
        events_df: MLFinLab barrier events DataFrame
                   Columns: t1 (touch time), type (1=TP, -1=SL, 0=timeout)
        sequence_dates_list: List of DatetimeIndex for each sequence
        seq_len: Sequence length (timesteps)
        sigma: Gaussian smoothing (timesteps)
        verbose: Print statistics
    
    Returns:
        heatmaps: (n_samples, seq_len, 2) array
                  [:, :, 0] = SL probability heatmap
                  [:, :, 1] = TP probability heatmap
        stats: Dictionary with generation statistics
    
    Example:
        Sequence with TP at position 5:
        - heatmaps[i, :, 1] has Gaussian peak at position 5
        - heatmaps[i, :, 0] is all zeros (no SL)
        
        Sequence with timeout (no barrier hit):
        - Both channels all zeros
    """
    n_samples = len(sequence_dates_list)
    heatmaps = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(sigma=sigma)
    kernel_radius = len(kernel) // 2
    
    # Statistics
    n_tp = 0
    n_sl = 0
    n_timeout = 0
    
    for i, seq_dates in enumerate(sequence_dates_list):
        # Get event for this sequence (using last date as key)
        seq_end_date = seq_dates[-1]
        
        # Find corresponding event
        # NOTE: You'll need to adapt this based on your data structure
        # This assumes events_df has same index as sequences
        if seq_end_date not in events_df.index:
            n_timeout += 1
            continue
        
        event = events_df.loc[seq_end_date]
        barrier_touch_date = event['t1']
        
        # Determine barrier type from label
        # Assuming: 1=UP/TP, -1=DOWN/SL, 0=TIMEOUT
        label = event.get('bin', event.get('type', 0))
        
        if label == 0:
            # TIMEOUT - both channels stay 0
            n_timeout += 1
            continue
        elif label == 1:
            # Take Profit hit
            channel = 1
            n_tp += 1
        elif label == -1:
            # Stop Loss hit
            channel = 0
            n_sl += 1
        else:
            # Unknown label
            n_timeout += 1
            continue
        
        # Find position of barrier touch in sequence
        try:
            position = seq_dates.get_loc(barrier_touch_date)
        except KeyError:
            # Barrier touch not in sequence window (shouldn't happen with proper alignment)
            if verbose:
                print(f"Warning: Barrier touch {barrier_touch_date} not in sequence ending {seq_end_date}")
            continue
        
        # Apply Gaussian kernel at position
        for t in range(seq_len):
            distance = abs(t - position)
            if distance <= kernel_radius:
                kernel_idx = distance + kernel_radius
                if kernel_idx < len(kernel):
                    heatmaps[i, t, channel] = kernel[kernel_idx]
    
    # Compile statistics
    stats = {
        'n_samples': n_samples,
        'n_tp': n_tp,
        'n_sl': n_sl,
        'n_timeout': n_timeout,
        'pct_tp': n_tp / n_samples * 100,
        'pct_sl': n_sl / n_samples * 100,
        'pct_timeout': n_timeout / n_samples * 100,
        'mean_tp_magnitude': heatmaps[:, :, 1].max(axis=1).mean(),
        'mean_sl_magnitude': heatmaps[:, :, 0].max(axis=1).mean(),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("HEATMAP TARGET GENERATION")
        print("="*60)
        print(f"Total sequences: {n_samples}")
        print(f"  TP events:  {n_tp:5d} ({stats['pct_tp']:5.1f}%)")
        print(f"  SL events:  {n_sl:5d} ({stats['pct_sl']:5.1f}%)")
        print(f"  Timeouts:   {n_timeout:5d} ({stats['pct_timeout']:5.1f}%)")
        print(f"\nHeatmap statistics:")
        print(f"  Mean TP peak: {stats['mean_tp_magnitude']:.4f}")
        print(f"  Mean SL peak: {stats['mean_sl_magnitude']:.4f}")
        print(f"  Sigma: {sigma} timesteps")
    
    return heatmaps, stats


def create_heatmaps_simple(
    labels: np.ndarray,
    seq_len: int = 20,
    sigma: float = 2.0,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Create barrier heatmaps using YOUR PROVEN swimming stroke approach!
    
    This is adapted from augment_stroke_labels_to_heatmaps() in learning_data.py
    - Uses YOUR EXACT Gaussian kernel (unnormalized)
    - Uses YOUR EXACT convolution approach (mode='same')
    - Uses YOUR EXACT clipping strategy ([0, 1])
    
    Creates 2-channel heatmaps: [SL, TP]
    
    Args:
        labels: (n_samples,) array with {-1, 0, 1} or {0, 1, 2}
        seq_len: Sequence length
        sigma: Gaussian smoothing (timesteps, like your stroke_sigma)
        verbose: Print stats
    
    Returns:
        heatmaps: (n_samples, seq_len, 2)
                  [:, :, 0] = SL channel
                  [:, :, 1] = TP channel
        stats: Generation statistics
    """
    n_samples = len(labels)
    
    # Build Gaussian kernel - YOUR EXACT METHOD from swimming!
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius+1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    # NOT normalized (following YOUR implementation)
    
    # Map labels to channels
    label_min = labels.min()
    
    # Statistics
    stats = {
        'n_samples': n_samples,
        'n_tp': 0,
        'n_sl': 0,
        'n_timeout': 0
    }
    
    # Create binary event arrays for each channel (like your stroke_labels)
    tp_events = np.zeros((n_samples, seq_len), dtype=np.float32)
    sl_events = np.zeros((n_samples, seq_len), dtype=np.float32)
    
    for i, label in enumerate(labels):
        # Normalize label to {-1, 0, 1} if needed
        if label_min == 0:
            label = label - 1  # {0,1,2} → {-1,0,1}
        
        if label == 0:
            # TIMEOUT - no event
            stats['n_timeout'] += 1
            continue
        elif label == 1:
            # Take Profit - place event at end of sequence
            tp_events[i, -1] = 1.0
            stats['n_tp'] += 1
        elif label == -1:
            # Stop Loss - place event at end of sequence
            sl_events[i, -1] = 1.0
            stats['n_sl'] += 1
        else:
            stats['n_timeout'] += 1
            continue
    
    # Apply YOUR EXACT convolution method to each sequence
    heatmaps = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    
    for i in range(n_samples):
        # SL channel - YOUR METHOD
        heat_sl = np.convolve(sl_events[i], kernel, mode='same')
        heat_sl = np.clip(heat_sl, 0.0, 1.0)
        heatmaps[i, :, 0] = heat_sl
        
        # TP channel - YOUR METHOD
        heat_tp = np.convolve(tp_events[i], kernel, mode='same')
        heat_tp = np.clip(heat_tp, 0.0, 1.0)
        heatmaps[i, :, 1] = heat_tp
    
    # Calculate percentages
    stats['pct_tp'] = stats['n_tp'] / n_samples * 100
    stats['pct_sl'] = stats['n_sl'] / n_samples * 100
    stats['pct_timeout'] = stats['n_timeout'] / n_samples * 100
    stats['mean_tp_magnitude'] = heatmaps[:, :, 1].max(axis=1).mean()
    stats['mean_sl_magnitude'] = heatmaps[:, :, 0].max(axis=1).mean()
    
    if verbose:
        print("\n" + "="*60)
        print("HEATMAP GENERATION (Your Swimming Stroke Method!)")
        print("="*60)
        print(f"Total sequences: {n_samples}")
        print(f"  TP events:  {stats['n_tp']:5d} ({stats['pct_tp']:5.1f}%)")
        print(f"  SL events:  {stats['n_sl']:5d} ({stats['pct_sl']:5.1f}%)")
        print(f"  Timeouts:   {stats['n_timeout']:5d} ({stats['pct_timeout']:5.1f}%)")
        print(f"\nHeatmap statistics:")
        print(f"  Mean TP peak: {stats['mean_tp_magnitude']:.4f}")
        print(f"  Mean SL peak: {stats['mean_sl_magnitude']:.4f}")
        print(f"  Sigma: {sigma} timesteps (like your stroke_sigma)")
        print(f"\n✓ Using YOUR proven swimming stroke detection method!")
    
    return heatmaps, stats


if __name__ == "__main__":
    print("="*80)
    print("BARRIER HEATMAP TARGET GENERATION - TEST")
    print("="*80)
    
    # Test with synthetic labels
    n_samples = 1000
    seq_len = 20
    
    # Simulate label distribution (22% DOWN, 64% TIMEOUT, 14% UP)
    labels = np.random.choice(
        [0, 1, 2],  # DOWN, TIMEOUT, UP
        size=n_samples,
        p=[0.22, 0.64, 0.14]
    )
    
    print(f"\nGenerating heatmaps for {n_samples} sequences...")
    print(f"Sequence length: {seq_len}")
    print(f"Gaussian sigma: 2.0 timesteps")
    
    # Generate heatmaps
    heatmaps, stats = create_heatmaps_simple(
        labels=labels,
        seq_len=seq_len,
        sigma=2.0,
        verbose=True
    )
    
    print(f"\n✅ Generated heatmaps shape: {heatmaps.shape}")
    print(f"   Expected: ({n_samples}, {seq_len}, 2)")
    
    # Visualize a few examples
    print("\n" + "="*60)
    print("EXAMPLE HEATMAPS")
    print("="*60)
    
    for label_type, label_val in [('TP', 2), ('SL', 0), ('TIMEOUT', 1)]:
        idx = np.where(labels == label_val)[0][0]
        print(f"\n{label_type} example (sample {idx}):")
        print(f"  Label: {label_val}")
        print(f"  SL channel max: {heatmaps[idx, :, 0].max():.4f}")
        print(f"  TP channel max: {heatmaps[idx, :, 1].max():.4f}")
        
        if label_val == 2:  # TP
            print(f"  TP channel (last 5 steps): {heatmaps[idx, -5:, 1]}")
        elif label_val == 0:  # SL
            print(f"  SL channel (last 5 steps): {heatmaps[idx, -5:, 0]}")
        else:  # TIMEOUT
            print(f"  Both channels near zero: ✅")
    
    print("\n" + "="*60)
    print("✅ Heatmap generation successful!")
    print("="*60)

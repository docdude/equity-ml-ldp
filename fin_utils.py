"""
Financial ML Utilities
Modular components for training, inference, and evaluation
"""

import pickle
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod


class FeatureNormalizer(ABC):
    """
    Abstract base class for feature normalization strategies.
    Ensures consistent interface for training, inference, and evaluation.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer on training data.
        
        Args:
            X: Features array of shape (n_samples, n_timesteps, n_features)
               or (n_samples, n_features)
        
        Returns:
            self (for chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted normalizer.
        
        Args:
            X: Features array of shape (n_samples, n_timesteps, n_features)
               or (n_samples, n_features)
        
        Returns:
            Normalized features (same shape as input)
        """
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving/logging"""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save normalizer state to file"""
        pass
    
    @staticmethod
    @abstractmethod
    def load(filepath: str) -> 'FeatureNormalizer':
        """Load normalizer state from file"""
        pass


class RobustNormalizer(FeatureNormalizer):
    """
    RobustScaler-based normalization (median/IQR).
    Best for financial data with outliers.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self._is_fitted = False
        self._original_shape = None
    
    def fit(self, X: np.ndarray) -> 'RobustNormalizer':
        """Fit on training data"""
        # Handle sequences
        if X.ndim == 3:
            self._original_shape = X.shape
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
        
        # Fit scaler
        self.scaler.fit(X_flat)
        self._is_fitted = True
        
        print(f"✅ RobustNormalizer fitted")
        print(f"   Input shape: {X.shape}")
        print(f"   Features: {X_flat.shape[1]}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using fitted scaler"""
        if not self._is_fitted:
            raise ValueError("Normalizer not fitted! Call .fit() first.")
        
        # Handle sequences
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(n_samples, n_timesteps, n_features)
        else:
            return self.scaler.transform(X)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration"""
        return {
            'method': 'RobustScaler',
            'scaler_type': 'sklearn.preprocessing.RobustScaler',
            'fitted': self._is_fitted,
            'uses_median_iqr': True,
            'robust_to_outliers': True
        }
    
    def save(self, filepath: str):
        """Save normalizer"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Saved RobustNormalizer to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'RobustNormalizer':
        """Load normalizer"""
        normalizer = RobustNormalizer()
        with open(filepath, 'rb') as f:
            normalizer.scaler = pickle.load(f)
        normalizer._is_fitted = True
        print(f"✅ Loaded RobustNormalizer from: {filepath}")
        return normalizer


class StandardNormalizer(FeatureNormalizer):
    """
    StandardScaler-based normalization (mean/std z-score).
    Good for clean data without outliers.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'StandardNormalizer':
        """Fit on training data"""
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
        
        self.scaler.fit(X_flat)
        self._is_fitted = True
        
        print(f"✅ StandardNormalizer fitted (z-score)")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using fitted scaler"""
        if not self._is_fitted:
            raise ValueError("Normalizer not fitted! Call .fit() first.")
        
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(n_samples, n_timesteps, n_features)
        else:
            return self.scaler.transform(X)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'method': 'StandardScaler',
            'scaler_type': 'sklearn.preprocessing.StandardScaler',
            'fitted': self._is_fitted,
            'uses_mean_std': True,
            'sensitive_to_outliers': True
        }
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Saved StandardNormalizer to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'StandardNormalizer':
        normalizer = StandardNormalizer()
        with open(filepath, 'rb') as f:
            normalizer.scaler = pickle.load(f)
        normalizer._is_fitted = True
        print(f"✅ Loaded StandardNormalizer from: {filepath}")
        return normalizer


class MinMaxNormalizer(FeatureNormalizer):
    """
    MinMaxScaler-based normalization (scale to [0, 1]).
    Use only for bounded features like RSI, not recommended for financial returns.
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_range = feature_range
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxNormalizer':
        """Fit on training data"""
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
        
        self.scaler.fit(X_flat)
        self._is_fitted = True
        
        print(f"✅ MinMaxNormalizer fitted to range {self.feature_range}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using fitted scaler"""
        if not self._is_fitted:
            raise ValueError("Normalizer not fitted! Call .fit() first.")
        
        if X.ndim == 3:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(n_samples, n_timesteps, n_features)
        else:
            return self.scaler.transform(X)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'method': 'MinMaxScaler',
            'scaler_type': 'sklearn.preprocessing.MinMaxScaler',
            'fitted': self._is_fitted,
            'feature_range': self.feature_range,
            'very_sensitive_to_outliers': True,
            'warning': 'Not recommended for financial data with outliers'
        }
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Saved MinMaxNormalizer to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'MinMaxNormalizer':
        normalizer = MinMaxNormalizer()
        with open(filepath, 'rb') as f:
            normalizer.scaler = pickle.load(f)
        normalizer._is_fitted = True
        print(f"✅ Loaded MinMaxNormalizer from: {filepath}")
        return normalizer


class NoNormalizer(FeatureNormalizer):
    """
    Identity normalizer (no normalization).
    Use when features are already properly normalized at source.
    """
    
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'NoNormalizer':
        """No-op fit"""
        self._is_fitted = True
        print(f"✅ NoNormalizer: No normalization will be applied")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return input unchanged"""
        return X
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'method': 'None',
            'scaler_type': 'Identity',
            'fitted': self._is_fitted,
            'note': 'Features used as-is without normalization'
        }
    
    def save(self, filepath: str):
        """Save empty marker file"""
        with open(filepath, 'wb') as f:
            pickle.dump({'normalizer': 'none'}, f)
        print(f"💾 Saved NoNormalizer marker to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'NoNormalizer':
        normalizer = NoNormalizer()
        normalizer._is_fitted = True
        print(f"✅ Loaded NoNormalizer from: {filepath}")
        return normalizer


def get_normalizer(method: str = 'robust', **kwargs) -> FeatureNormalizer:
    """
    Factory function to get normalizer by name.
    
    Args:
        method: Normalization method ('robust', 'standard', 'minmax', 'none')
        **kwargs: Additional arguments for specific normalizers
    
    Returns:
        FeatureNormalizer instance
    
    Examples:
        >>> normalizer = get_normalizer('robust')
        >>> normalizer = get_normalizer('minmax', feature_range=(-1, 1))
        >>> normalizer = get_normalizer('none')
    """
    method = method.lower()
    
    if method in ['robust', 'robustscaler']:
        return RobustNormalizer()
    elif method in ['standard', 'standardscaler', 'zscore']:
        return StandardNormalizer()
    elif method in ['minmax', 'minmaxscaler']:
        return MinMaxNormalizer(**kwargs)
    elif method in ['none', 'identity', 'no', 'null']:
        return NoNormalizer()
    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Choose from: 'robust', 'standard', 'minmax', 'none'"
        )


def validate_normalization(X: np.ndarray, name: str = "Data", method: str = 'robust') -> Dict[str, float]:
    """
    Validate that normalization looks correct.
    
    Args:
        X: Normalized features
        name: Name for logging
        method: Normalization method used ('robust', 'standard', 'minmax', 'none')
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': float(X.mean()),
        'std': float(X.std()),
        'median': float(np.median(X)),
        'min': float(X.min()),
        'max': float(X.max()),
        'p25': float(np.percentile(X, 25)),
        'p75': float(np.percentile(X, 75)),
    }
    
    # Define expected ranges per normalization method
    method = method.lower()
    if method in ['minmax', 'minmaxscaler']:
        expected = {
            'mean': (0.3, 0.7),
            'std': (0.1, 0.9),
            'range': (-0.5, 1.5),  # Allow some overshoot
            'description': 'MinMaxScaler: data scaled to [0, 1]'
        }
    elif method in ['robust', 'robustscaler']:
        expected = {
            'mean': (-0.5, 0.5),
            'std': (0.5, 2.0),
            'range': (-10, 10),
            'description': 'RobustScaler: median=0, scaled by IQR'
        }
    elif method in ['standard', 'standardscaler', 'zscore']:
        expected = {
            'mean': (-0.3, 0.3),
            'std': (0.7, 1.3),
            'range': (-10, 10),
            'description': 'StandardScaler: mean=0, std=1'
        }
    elif method in ['none', 'identity', 'no', 'null']:
        expected = {
            'mean': None,
            'std': None,
            'range': None,
            'description': 'No normalization applied'
        }
    else:
        expected = {
            'mean': None,
            'std': None,
            'range': None,
            'description': f'Unknown method: {method}'
        }
    
    print(f"\n🔍 {name} Normalization Validation:")
    print(f"   Method: {expected['description']}")
    print(f"   Shape: {X.shape}")
    print(f"   Mean: {stats['mean']:.4f}")
    print(f"   Std: {stats['std']:.4f}")
    print(f"   Median: {stats['median']:.4f}")
    print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"   IQR: [{stats['p25']:.4f}, {stats['p75']:.4f}]")
    
    # Check for issues
    issues = []
    
    if np.isnan(X).any():
        issues.append("❌ NaN values detected!")
    
    if np.isinf(X).any():
        issues.append("❌ Inf values detected!")
    
    # Method-specific validation
    if expected['mean'] is not None:
        mean_min, mean_max = expected['mean']
        if not (mean_min <= stats['mean'] <= mean_max):
            issues.append(f"⚠️  Mean outside expected range: {stats['mean']:.4f} (expected {mean_min} to {mean_max})")
    
    if expected['std'] is not None:
        std_min, std_max = expected['std']
        if not (std_min <= stats['std'] <= std_max):
            issues.append(f"⚠️  Std outside expected range: {stats['std']:.4f} (expected {std_min} to {std_max})")
    
    if expected['range'] is not None:
        range_min, range_max = expected['range']
        if stats['min'] < range_min or stats['max'] > range_max:
            issues.append(f"⚠️  Extreme values: [{stats['min']:.2f}, {stats['max']:.2f}] (expected [{range_min}, {range_max}])")
    
    if issues:
        print("\n⚠️  Validation issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ Normalization looks good!")
    
    return stats


def normalize_sequences(
    X: np.ndarray,
    method: str = 'robust',
    normalizer: Optional[FeatureNormalizer] = None,
    fit: bool = True,
    validate: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, FeatureNormalizer]:
    """
    Normalize feature sequences with flexible options.
    
    Args:
        X: Features of shape (n_samples, n_timesteps, n_features)
        method: Normalization method if normalizer not provided
        normalizer: Pre-fitted normalizer (if provided, uses this)
        fit: Whether to fit normalizer (ignored if normalizer provided)
        validate: Whether to validate normalization quality
        verbose: Whether to print statistics
    
    Returns:
        Tuple of (normalized_features, normalizer)
    
    Examples:
        # Training: fit and transform
        X_train_scaled, normalizer = normalize_sequences(X_train, method='robust', fit=True)
        
        # Validation/Test: transform only (no fitting)
        X_val_scaled, _ = normalize_sequences(X_val, normalizer=normalizer, fit=False)
        
        # Inference: load and transform
        normalizer = RobustNormalizer.load('scaler.pkl')
        X_new_scaled, _ = normalize_sequences(X_new, normalizer=normalizer, fit=False)
    """
    # Get or create normalizer
    if normalizer is None:
        normalizer = get_normalizer(method)
    
    # Statistics before normalization
    if verbose:
        print(f"\n📊 Feature statistics BEFORE normalization:")
        print(f"   Mean: {X.mean():.4f}, Std: {X.std():.4f}")
        print(f"   Min: {X.min():.4f}, Max: {X.max():.4f}")
    
    # Handle NaN/Inf
    n_inf = np.isinf(X).sum()
    n_nan = np.isnan(X).sum()
    if n_inf > 0 or n_nan > 0:
        if verbose:
            print(f"   ⚠️  WARNING: {n_inf} inf, {n_nan} NaN values")
            print(f"   ✅ Replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize
    if fit and not normalizer._is_fitted:
        X_scaled = normalizer.fit_transform(X)
    else:
        X_scaled = normalizer.transform(X)
    
    # Statistics after normalization
    if verbose:
        print(f"\n📊 Feature statistics AFTER normalization:")
        print(f"   Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
        print(f"   Min: {X_scaled.min():.4f}, Max: {X_scaled.max():.4f}")
    
    # Validate (infer method from normalizer type if not provided)
    if validate and verbose:
        # Get method from normalizer config
        config = normalizer.get_config()
        normalizer_method = config.get('method', method).lower()
        validate_normalization(X_scaled, name="Normalized features", method=normalizer_method)
    
    return X_scaled, normalizer


# Quick access functions for common use cases

def normalize_for_training(
    X_train: np.ndarray,
    X_val: np.ndarray,
    method: str = 'robust',
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, FeatureNormalizer]:
    """
    Normalize training and validation data (fit on train only).
    
    Args:
        X_train: Training features
        X_val: Validation features
        method: Normalization method
        save_path: Path to save normalizer (optional)
    
    Returns:
        (X_train_scaled, X_val_scaled, normalizer)
    """
    print(f"\n{'='*60}")
    print(f"NORMALIZING DATA FOR TRAINING (FIT ON TRAIN ONLY)")
    print(f"{'='*60}")
    
    # Normalize training data (fit and transform)
    print(f"\n1. Training data:")
    X_train_scaled, normalizer = normalize_sequences(
        X_train, method=method, fit=True, validate=True
    )
    
    # Normalize validation data (transform only, no fitting)
    print(f"\n2. Validation data:")
    X_val_scaled, _ = normalize_sequences(
        X_val, normalizer=normalizer, fit=False, validate=True
    )
    
    # Save normalizer
    if save_path:
        normalizer.save(save_path)
    
    return X_train_scaled, X_val_scaled, normalizer


def normalize_for_inference(
    X: np.ndarray,
    normalizer_path: str
) -> np.ndarray:
    """
    Normalize features for inference using saved normalizer.
    
    Args:
        X: Features to normalize
        normalizer_path: Path to saved normalizer
    
    Returns:
        Normalized features
    """
    # Try to detect normalizer type from file
    with open(normalizer_path, 'rb') as f:
        obj = pickle.load(f)
    
    # Determine type and load appropriately
    if isinstance(obj, dict) and obj.get('normalizer') == 'none':
        normalizer = NoNormalizer.load(normalizer_path)
    elif hasattr(obj, 'center_'):  # RobustScaler
        normalizer = RobustNormalizer.load(normalizer_path)
    elif hasattr(obj, 'mean_'):  # StandardScaler
        normalizer = StandardNormalizer.load(normalizer_path)
    elif hasattr(obj, 'min_'):  # MinMaxScaler
        normalizer = MinMaxNormalizer.load(normalizer_path)
    else:
        raise ValueError(f"Unknown normalizer type in {normalizer_path}")
    
    # Transform
    X_scaled, _ = normalize_sequences(X, normalizer=normalizer, fit=False)
    return X_scaled


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("FEATURE NORMALIZER DEMO")
    print("="*60)
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 20, 10) * 50 + 100  # (samples, timesteps, features)
    X_val = np.random.randn(20, 20, 10) * 50 + 100
    X_test = np.random.randn(20, 20, 10) * 50 + 100
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Method 1: Using high-level function
    print(f"\n{'='*60}")
    print("METHOD 1: High-level function (recommended)")
    print(f"{'='*60}")
    
    X_train_norm, X_val_norm, normalizer = normalize_for_training(
        X_train, X_val, method='robust', save_path='/tmp/test_normalizer.pkl'
    )
    
    # Test inference
    print(f"\n{'='*60}")
    print("INFERENCE TEST")
    print(f"{'='*60}")
    X_test_norm = normalize_for_inference(X_test, '/tmp/test_normalizer.pkl')
    
    # Method 2: Manual control (for custom workflows)
    print(f"\n{'='*60}")
    print("METHOD 2: Manual control (advanced)")
    print(f"{'='*60}")
    
    normalizer2 = get_normalizer('standard')
    X_train_norm2 = normalizer2.fit_transform(X_train)
    X_val_norm2 = normalizer2.transform(X_val)
    
    print(f"\n✅ Demo complete!")

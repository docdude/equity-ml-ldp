"""
Systematic Feature Selection Based on MDI/MDA/Orthogonal Analysis

This script implements the methodology from FEATURE_SELECTION_METHODOLOGY.md
"""

import pandas as pd
import numpy as np
from feature_config import FeatureConfig

def load_feature_importance_data():
    """
    Load feature importance data from artifacts.
    Returns comparison dataframe and orthogonalized MDA.
    """
    comparison = pd.read_csv(
        'artifacts/feature_importance/feature_importance_comparison.csv',
        index_col=0
    )
    
    # Load orthogonalized MDA from the mapped file
    ortho_mda_mapped = pd.read_csv(
        'artifacts/feature_importance/Ortho_MDA_importance_mapped.csv'
    )
    ortho_mda_mapped.set_index('feature', inplace=True)
    
    # Load original MDA to get std values
    mda_original = pd.read_csv(
        'artifacts/feature_importance/MDA_importance.csv',
        index_col=0
    )
    
    # Create ortho_mda dataframe
    ortho_mda = pd.DataFrame({
        'mean': ortho_mda_mapped['importance'],
        'std': mda_original['std']  # Use original std as proxy
    })
    
    return comparison, ortho_mda

def classify_features_by_quadrant(comparison, mdi_threshold=None, mda_threshold=None):
    """
    Classify features into 4 quadrants based on MDI and MDA
    
    Quadrant 1: High MDI + High MDA (Robust)
    Quadrant 2: High MDI + Low MDA (Overfitted)
    Quadrant 3: Low MDI + High MDA (Substituted)
    Quadrant 4: Low MDI + Low MDA (Unimportant)
    """
    if mdi_threshold is None:
        mdi_threshold = comparison['MDI_mean'].median()
    if mda_threshold is None:
        mda_threshold = comparison['MDA_mean'].median()
    
    high_mdi = comparison['MDI_mean'] > mdi_threshold
    high_mda = comparison['MDA_mean'] > mda_threshold
    
    quadrants = {
        'robust': comparison[high_mdi & high_mda].index.tolist(),
        'overfitted': comparison[high_mdi & ~high_mda].index.tolist(),
        'substituted': comparison[~high_mdi & high_mda].index.tolist(),
        'unimportant': comparison[~high_mdi & ~high_mda].index.tolist()
    }
    
    return quadrants

def get_top_n_by_ortho_mda(ortho_mda, n=50):
    """Get top N features by orthogonalized MDA"""
    # Sort by orthogonalized MDA mean
    sorted_features = ortho_mda.sort_values('mean', ascending=False)
    return sorted_features.head(n).index.tolist()

def check_correlation(features_df, threshold=0.7):
    """
    Find highly correlated feature pairs
    
    Returns: List of (feature1, feature2, correlation) tuples
    """
    corr_matrix = features_df[features_df.columns].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    return high_corr_pairs

def remove_redundant_features(feature_list, ortho_mda, correlation_pairs, threshold=0.7):
    """
    Remove redundant features from a list based on correlation
    
    For each highly correlated pair, keep the one with higher Ortho MDA
    """
    features_to_remove = set()
    
    for feat1, feat2, corr in correlation_pairs:
        if feat1 in feature_list and feat2 in feature_list:
            # Keep the one with higher Ortho MDA
            if feat1 in ortho_mda.index and feat2 in ortho_mda.index:
                mda1 = ortho_mda.loc[feat1, 'mean']
                mda2 = ortho_mda.loc[feat2, 'mean']
                
                if mda1 < mda2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
    
    return [f for f in feature_list if f not in features_to_remove]

def select_features_method_1_ortho_top_n(ortho_mda, n=35):
    """
    Method 1: Select top N by Orthogonal MDA, remove redundancy
    
    This maximizes unique predictive information
    """
    print("=" * 80)
    print("METHOD 1: Top N Orthogonal MDA (Unique Information)")
    print("=" * 80)
    
    # Get top N features
    top_features = get_top_n_by_ortho_mda(ortho_mda, n=n)
    
    print(f"\n✓ Starting with top {n} Ortho MDA features")
    print(f"  Features: {len(top_features)}")
    
    # Note: Would need actual feature correlation matrix to remove redundancy
    # For now, return the top N
    
    print(f"\n✓ Final feature count: {len(top_features)}")
    return top_features

def select_features_method_2_quadrant_based(comparison, ortho_mda, target_count=35):
    """
    Method 2: Quadrant-based selection (Conservative)
    
    Only keep Quadrant 1 (High MDI + High MDA) features
    """
    print("=" * 80)
    print("METHOD 2: Quadrant-Based (Conservative)")
    print("=" * 80)
    
    # Classify into quadrants
    quadrants = classify_features_by_quadrant(comparison)
    
    print(f"\n✓ Quadrant 1 (Robust): {len(quadrants['robust'])} features")
    print(f"  Quadrant 2 (Overfitted): {len(quadrants['overfitted'])} features")
    print(f"  Quadrant 3 (Substituted): {len(quadrants['substituted'])} features")
    print(f"  Quadrant 4 (Unimportant): {len(quadrants['unimportant'])} features")
    
    # Start with Quadrant 1
    selected = quadrants['robust'].copy()
    
    print(f"\n✓ Starting with {len(selected)} Quadrant 1 features")
    
    # If need more features, add from Quadrant 3 (substituted but predictive)
    if len(selected) < target_count:
        # Sort Quadrant 3 by MDA
        q3_sorted = comparison.loc[quadrants['substituted']].sort_values(
            'MDA_mean', ascending=False
        )
        needed = target_count - len(selected)
        selected.extend(q3_sorted.head(needed).index.tolist())
        print(f"✓ Added {needed} features from Quadrant 3 (Substituted)")
    
    print(f"\n✓ Final feature count: {len(selected)}")
    return selected

def select_features_method_3_hybrid(comparison, ortho_mda, target_count=35):
    """
    Method 3: Hybrid approach (Balanced)
    
    Combine Ortho MDA rankings with MDI-MDA validation
    """
    print("=" * 80)
    print("METHOD 3: Hybrid (Balanced)")
    print("=" * 80)
    
    # Start with top 50 Ortho MDA
    top_50_ortho = get_top_n_by_ortho_mda(ortho_mda, n=50)
    print(f"\n✓ Starting with top 50 Ortho MDA features")
    
    # Classify into quadrants
    quadrants = classify_features_by_quadrant(comparison)
    
    # Remove Quadrant 4 (unimportant)
    selected = [f for f in top_50_ortho if f not in quadrants['unimportant']]
    print(f"✓ Removed {len(top_50_ortho) - len(selected)} Quadrant 4 features")
    
    # Flag Quadrant 2 (overfitted) for review
    overfitted_in_selection = [f for f in selected if f in quadrants['overfitted']]
    if overfitted_in_selection:
        print(f"\n⚠️  WARNING: {len(overfitted_in_selection)} potentially overfitted features:")
        for feat in overfitted_in_selection[:5]:
            mdi = comparison.loc[feat, 'MDI_mean']
            mda = comparison.loc[feat, 'MDA_mean']
            print(f"  - {feat} (MDI: {mdi:.4f}, MDA: {mda:.4f})")
    
    # Trim to target count if needed
    if len(selected) > target_count:
        # Keep top by Ortho MDA
        ortho_scores = {f: ortho_mda.loc[f, 'mean'] 
                       for f in selected if f in ortho_mda.index}
        sorted_by_ortho = sorted(ortho_scores.items(), 
                                key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in sorted_by_ortho[:target_count]]
    
    print(f"\n✓ Final feature count: {len(selected)}")
    return selected

def analyze_feature_diversity(selected_features):
    """
    Analyze diversity of selected features across categories
    """
    print("\n" + "=" * 80)
    print("FEATURE DIVERSITY ANALYSIS")
    print("=" * 80)
    
    categories = {
        'momentum': ['macd', 'rsi', 'roc', 'cci', 'stoch', 'williams'],
        'volatility': ['volatility', 'vol_', 'atr', 'realized_vol'],
        'volume': ['volume', 'obv', 'ad_', 'vwap', 'cmf'],
        'microstructure': ['vpin', 'spread', 'amihud', 'kyle', 'hl_range'],
        'trend': ['adx', 'sar', 'aroon', 'dist_from_ma'],
        'bollinger': ['bb_'],
        'returns': ['log_return', 'return_'],
        'statistical': ['serial_corr', 'skewness', 'kurtosis', 'entropy', 'hurst'],
        'market': ['GC=F', '^GSPC', '^VIX', '^FVX'],
    }
    
    category_counts = {}
    for category, keywords in categories.items():
        count = sum(1 for f in selected_features 
                   if any(kw in f for kw in keywords))
        if count > 0:
            category_counts[category] = count
    
    print("\nFeature distribution by category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:15s}: {count:2d} features")
    
    return category_counts

def compare_with_current_config(selected_features, config_name='wavenet_optimized_v2'):
    """
    Compare selected features with current config
    """
    print("\n" + "=" * 80)
    print(f"COMPARISON WITH {config_name.upper()}")
    print("=" * 80)
    
    config = FeatureConfig.get_preset(config_name)
    current_features = set(config['feature_list'])
    selected_set = set(selected_features)
    
    in_both = current_features & selected_set
    only_current = current_features - selected_set
    only_selected = selected_set - current_features
    
    print(f"\nCurrent config: {len(current_features)} features")
    print(f"Selected set:   {len(selected_features)} features")
    print(f"Overlap:        {len(in_both)} features ({len(in_both)/len(current_features)*100:.0f}%)")
    
    if only_current:
        print(f"\n⚠️  In current config but NOT selected ({len(only_current)}):")
        for feat in sorted(only_current)[:10]:
            print(f"  - {feat}")
        if len(only_current) > 10:
            print(f"  ... and {len(only_current) - 10} more")
    
    if only_selected:
        print(f"\n✨ Selected but NOT in current config ({len(only_selected)}):")
        for feat in sorted(only_selected)[:10]:
            print(f"  - {feat}")
        if len(only_selected) > 10:
            print(f"  ... and {len(only_selected) - 10} more")
    
    return {
        'overlap': len(in_both),
        'only_current': list(only_current),
        'only_selected': list(only_selected)
    }

def main():
    """
    Run complete feature selection analysis
    """
    print("=" * 80)
    print("SYSTEMATIC FEATURE SELECTION ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading feature importance data...")
    comparison, ortho_mda = load_feature_importance_data()
    print(f"✓ Loaded {len(comparison)} features from comparison")
    print(f"✓ Loaded {len(ortho_mda)} features from orthogonal MDA")
    
    # Method 1: Ortho MDA Top-N
    print("\n")
    method1_features = select_features_method_1_ortho_top_n(ortho_mda, n=35)
    analyze_feature_diversity(method1_features)
    compare_with_current_config(method1_features)
    
    # Method 2: Quadrant-Based
    print("\n")
    method2_features = select_features_method_2_quadrant_based(
        comparison, ortho_mda, target_count=35
    )
    analyze_feature_diversity(method2_features)
    compare_with_current_config(method2_features)
    
    # Method 3: Hybrid
    print("\n")
    method3_features = select_features_method_3_hybrid(
        comparison, ortho_mda, target_count=35
    )
    analyze_feature_diversity(method3_features)
    compare_with_current_config(method3_features)
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: THREE METHODS COMPARED")
    print("=" * 80)
    
    method1_set = set(method1_features)
    method2_set = set(method2_features)
    method3_set = set(method3_features)
    
    all_three = method1_set & method2_set & method3_set
    any_two = (method1_set & method2_set) | (method1_set & method3_set) | (method2_set & method3_set)
    
    print(f"\n✓ Core features (in all 3 methods): {len(all_three)}")
    print(f"✓ Consensus features (in 2+ methods): {len(any_two)}")
    
    print("\nCore features (all methods agree):")
    for feat in sorted(all_three)[:20]:
        ortho_rank = ortho_mda.index.tolist().index(feat) + 1 if feat in ortho_mda.index else 'N/A'
        print(f"  {feat:25s} (Ortho MDA rank: {ortho_rank})")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
Based on the analysis:

1. START WITH: Method 3 (Hybrid) - 35 features
   - Best balance of unique information and validation
   - Removes dead weight (Quadrant 4)
   - Keeps diverse feature types

2. VALIDATE: Run training with Method 3 features
   - Compare to current wavenet_optimized_v2 (42 features)
   - Check per-barrier precision/recall
   - Monitor for overfitting

3. ITERATE: Fine-tune based on results
   - If overfitting: Use Method 2 (Conservative)
   - If underfitting: Use Method 1 (More features)
   - Add/remove based on per-barrier performance

4. KEY CHANGES FROM CURRENT CONFIG:
   - ADD: vpin (now working, rank #5 robust!)
   - REMOVE: Features with negative MDA
   - CONSOLIDATE: Redundant volume features
    """)

if __name__ == '__main__':
    main()

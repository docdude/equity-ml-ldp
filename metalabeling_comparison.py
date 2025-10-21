"""
Visual Comparison: Standard Triple Barrier vs Meta-Labeling
===========================================================

Run this to see the difference between the two approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_difference():
    """Show the key difference between standard and meta-labeling"""
    
    print("="*80)
    print("STANDARD TRIPLE BARRIER vs META-LABELING")
    print("="*80)
    
    # Create example data
    dates = pd.date_range('2024-01-01', periods=10)
    
    print("\n" + "="*80)
    print("SCENARIO 1: STANDARD TRIPLE BARRIER (What we're doing)")
    print("="*80)
    
    # Standard approach
    standard_events = pd.DataFrame({
        'date': dates,
        't1': dates + pd.Timedelta(days=7),
        'trgt': [0.02] * 10,
        'pt': [2.0] * 10,
        'sl': [1.0] * 10,
        # NO 'side' column - this is the key!
    })
    
    standard_labels = pd.DataFrame({
        'date': dates,
        'ret': [0.015, -0.012, 0.008, -0.018, 0.022, -0.009, 0.003, 0.019, -0.015, 0.011],
        'trgt': [0.02] * 10,
        'bin': [1, -1, 0, -1, 1, -1, 0, 1, -1, 1]  # 3 classes!
    })
    
    print("\nEvents DataFrame (no 'side' column):")
    print(standard_events)
    
    print("\nLabels DataFrame:")
    print(standard_labels)
    
    print("\nLabel Distribution:")
    print(standard_labels['bin'].value_counts().sort_index())
    print("\nInterpretation:")
    print("  -1: PREDICTION = Price will go DOWN (hit stop-loss)")
    print("   0: PREDICTION = Price will TIMEOUT (hit vertical barrier)")
    print("   1: PREDICTION = Price will go UP (hit take-profit)")
    print("\n→ This is DIRECTION prediction (3-class problem)")
    print("→ WaveNet learns: What will price do next?")
    
    print("\n" + "="*80)
    print("SCENARIO 2: TRUE META-LABELING (López de Prado's way)")
    print("="*80)
    
    # Meta-labeling approach
    meta_events = pd.DataFrame({
        'date': dates,
        't1': dates + pd.Timedelta(days=7),
        'trgt': [0.02] * 10,
        'pt': [2.0] * 10,
        'sl': [2.0] * 10,  # Symmetric when using side
        'side': [1, 1, -1, 1, -1, 1, -1, -1, 1, -1]  # ✅ Has 'side' column!
    })
    
    meta_labels = pd.DataFrame({
        'date': dates,
        'ret': [0.015, -0.012, -0.008, -0.018, -0.022, 0.009, -0.003, -0.019, 0.015, -0.011],
        'trgt': [0.02] * 10,
        'side': [1, 1, -1, 1, -1, 1, -1, -1, 1, -1],
        'bin': [1, 0, 1, 0, 1, 1, 0, 1, 1, 1]  # 2 classes only!
    })
    
    print("\nEvents DataFrame (WITH 'side' column):")
    print(meta_events)
    
    print("\nMeta-Labels DataFrame:")
    print(meta_labels)
    
    print("\nLabel Distribution:")
    print(meta_labels['bin'].value_counts().sort_index())
    print("\nInterpretation:")
    print("   0: PREDICTION = Don't take this bet (false positive, or too risky)")
    print("   1: PREDICTION = Take this bet (confident in primary signal)")
    print("\n→ This is BET SIZING (2-class problem)")
    print("→ WaveNet learns: Should we trust the primary model's signal?")
    print("→ Primary model already said BUY or SELL (in 'side' column)")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Aspect': [
            'side_prediction parameter',
            'side column in events',
            'Number of classes',
            'Label values',
            'Model predicts',
            'Primary model needed?',
            'Use case',
            'Complexity'
        ],
        'Standard Triple Barrier': [
            'None',
            'No',
            '3 (or 2 if no timeout)',
            '{-1, 0, 1}',
            'Direction (UP/DOWN/NEUTRAL)',
            'No',
            'End-to-end direction prediction',
            'Single model, harder problem'
        ],
        'Meta-Labeling': [
            'Required (primary signals)',
            'Yes',
            '2',
            '{0, 1}',
            'Confidence (TAKE/SKIP)',
            'Yes (generates side)',
            'Filter primary model signals',
            'Two models, easier problem'
        ]
    })
    
    print(comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("WHAT HAPPENS IN CODE")
    print("="*80)
    
    print("\n1. Standard Triple Barrier (Our current approach):")
    print("-" * 60)
    print("""
    events = get_events(
        close=prices,
        t_events=dates,
        pt_sl=[2, 1],
        target=volatility,
        side_prediction=None  # ← NO PRIMARY MODEL
    )
    # events has NO 'side' column
    
    labels = meta_labeling(events, prices)
    # Returns: {-1, 0, 1} = direction labels
    
    # Train model:
    X_seq → y = {-1, 0, 1}  # Predict direction
    """)
    
    print("\n2. True Meta-Labeling (López de Prado):")
    print("-" * 60)
    print("""
    # First: Get primary signals
    primary_signals = create_primary_model(prices)  # {-1, 1}
    
    events = get_events(
        close=prices,
        t_events=signal_dates,
        pt_sl=[2, 2],  # Symmetric
        target=volatility,
        side_prediction=primary_signals  # ← HAS PRIMARY MODEL
    )
    # events HAS 'side' column
    
    labels = meta_labeling(events, prices)
    # Returns: {0, 1} = bet sizing labels
    
    # Train model:
    X_seq → y = {0, 1}  # Predict bet size/confidence
    """)
    
    print("\n" + "="*80)
    print("WHY THIS MATTERS FOR OUR PROJECT")
    print("="*80)
    
    print("""
1. We thought we were using meta-labeling
   → We're actually using standard triple-barrier
   → The function name is misleading!

2. Our WaveNet is learning DIRECTION
   → Not bet sizing
   → 3-class problem (harder)
   → Explains 100% neutral predictions

3. The 100% neutral issue is from:
   → Low signal-to-noise in direction prediction
   → Not from incorrect meta-labeling
   → Because we're not using meta-labeling!

4. To fix:
   → Option A: Keep direction prediction, fix with heatmaps
   → Option B: Add primary model, use true meta-labeling

5. Heatmap approach is still valid:
   → Works for direction prediction
   → Augment labels before sequencing
   → Should reduce neutral bias
    """)

def visualize_label_flow():
    """Visualize how labels flow through each approach"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Standard Triple Barrier
    ax1 = axes[0]
    ax1.text(0.5, 0.9, 'STANDARD TRIPLE BARRIER', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             transform=ax1.transAxes)
    
    # Flow chart
    boxes = [
        (0.15, 0.7, 'OHLCV Data'),
        (0.15, 0.5, 'Triple Barriers\n(no side)'),
        (0.15, 0.3, 'meta_labeling()'),
        (0.15, 0.1, 'Labels:\n{-1, 0, 1}'),
        (0.5, 0.1, 'WaveNet'),
        (0.85, 0.1, 'Predict:\nDIRECTION')
    ]
    
    for x, y, text in boxes:
        ax1.add_patch(plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1, 
                                     fill=True, facecolor='lightblue', 
                                     edgecolor='black', linewidth=2,
                                     transform=ax1.transAxes))
        ax1.text(x, y, text, ha='center', va='center', fontsize=10,
                transform=ax1.transAxes)
    
    # Arrows
    arrows = [(0.15, 0.65), (0.15, 0.45), (0.15, 0.25), (0.23, 0.1), (0.58, 0.1)]
    for x, y in arrows:
        ax1.annotate('', xy=(x, y-0.05), xytext=(x, y+0.05),
                    arrowprops=dict(arrowstyle='->', lw=2),
                    transform=ax1.transAxes)
    
    ax1.text(0.85, 0.5, '3-class problem\nHarder to learn\nLow signal/noise',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             transform=ax1.transAxes)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Meta-Labeling
    ax2 = axes[1]
    ax2.text(0.5, 0.9, 'TRUE META-LABELING', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             transform=ax2.transAxes)
    
    # Flow chart
    boxes = [
        (0.15, 0.7, 'OHLCV Data'),
        (0.15, 0.5, 'Primary Model\n(indicators)'),
        (0.15, 0.3, 'side_prediction\n{-1, 1}'),
        (0.5, 0.5, 'Triple Barriers\n(with side)'),
        (0.5, 0.3, 'meta_labeling()'),
        (0.5, 0.1, 'Labels:\n{0, 1}'),
        (0.85, 0.1, 'WaveNet'),
        (0.85, 0.3, 'Predict:\nBET SIZE')
    ]
    
    for x, y, text in boxes:
        color = 'lightgreen' if 'side' in text.lower() else 'lightblue'
        ax2.add_patch(plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1, 
                                     fill=True, facecolor=color, 
                                     edgecolor='black', linewidth=2,
                                     transform=ax2.transAxes))
        ax2.text(x, y, text, ha='center', va='center', fontsize=10,
                transform=ax2.transAxes)
    
    # Arrows
    arrows = [
        (0.15, 0.65), (0.15, 0.45), (0.23, 0.3), (0.42, 0.5), 
        (0.5, 0.45), (0.5, 0.25), (0.5, 0.15), (0.85, 0.15)
    ]
    for x, y in arrows:
        if y == 0.3 and x == 0.23:
            ax2.annotate('', xy=(0.42, 0.48), xytext=(0.23, 0.32),
                        arrowprops=dict(arrowstyle='->', lw=2),
                        transform=ax2.transAxes)
        else:
            ax2.annotate('', xy=(x, y-0.05), xytext=(x, y+0.05),
                        arrowprops=dict(arrowstyle='->', lw=2),
                        transform=ax2.transAxes)
    
    ax2.text(0.85, 0.6, '2-class problem\nEasier to learn\nFilters noise',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
             transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('metalabeling_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved diagram to: metalabeling_comparison.png")
    plt.show()

if __name__ == "__main__":
    demonstrate_difference()
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)
    visualize_label_flow()

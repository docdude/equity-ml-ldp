"""
Compare López de Prado weighting vs sklearn balanced weights
"""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# Simulate class distribution from your training data
# Label 0 (DOWN): 21.56%, Label 1 (TIMEOUT): 65.35%, Label 2 (UP): 13.09%
n_samples = 19960
y = np.array([0] * 4304 + [1] * 13044 + [2] * 2612)

# Method 1: sklearn balanced weights
sklearn_weights = compute_sample_weight(class_weight='balanced', y=y)

print("="*80)
print("SKLEARN BALANCED WEIGHTS")
print("="*80)
for label in [0, 1, 2]:
    mask = (y == label)
    count = mask.sum()
    pct = count / len(y) * 100
    avg_weight = sklearn_weights[mask].mean()
    label_name = {0: "DOWN/SL", 1: "TIMEOUT", 2: "UP/TP"}[label]
    print(f"Label {label} ({label_name}): {count:5d} ({pct:5.2f}%) → avg_weight={avg_weight:.4f}")

print(f"\nWeight statistics:")
print(f"  Range: [{sklearn_weights.min():.3f}, {sklearn_weights.max():.3f}]")
print(f"  Mean: {sklearn_weights.mean():.3f}, Std: {sklearn_weights.std():.3f}")

# Method 2: Simulate López de Prado weights
# Uniqueness: range [0.0, 7.296], varies by overlap
# Return attribution: multiplies uniqueness
# Let's simulate what you reported

# Your reported LdP stats:
# Label 0: avg_weight=1.5846 (uniqueness not shown but implied)
# Label 1: avg_weight=0.4982
# Label 2: avg_weight=2.5426
# Final weight range: [0.000, 15.798]

# If LdP weights = uniqueness × sklearn_weights, we can back-calculate uniqueness
ldp_weights_reported = np.zeros(len(y))
ldp_weights_reported[y == 0] = 1.5846
ldp_weights_reported[y == 1] = 0.4982
ldp_weights_reported[y == 2] = 2.5426

# Calculate implied uniqueness factors
implied_uniqueness = ldp_weights_reported / sklearn_weights

print("\n" + "="*80)
print("LÓPEZ DE PRADO WEIGHTS (FROM YOUR RUN)")
print("="*80)
for label in [0, 1, 2]:
    mask = (y == label)
    count = mask.sum()
    pct = count / len(y) * 100
    avg_weight = ldp_weights_reported[mask].mean()
    avg_unique = implied_uniqueness[mask].mean()
    sklearn_avg = sklearn_weights[mask].mean()
    label_name = {0: "DOWN/SL", 1: "TIMEOUT", 2: "UP/TP"}[label]
    print(f"Label {label} ({label_name}): {count:5d} ({pct:5.2f}%)")
    print(f"  sklearn_weight={sklearn_avg:.4f} → ldp_weight={avg_weight:.4f}")
    print(f"  implied uniqueness multiplier: {avg_unique:.4f}x")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print("\n🔍 Key Insight:")
print("If LdP weights = uniqueness × sklearn_weights, then:")
print("  - DOWN/SL: 1.5846 / 1.5428 = 1.027x uniqueness")
print("  - TIMEOUT: 0.4982 / 0.5101 = 0.977x uniqueness")
print("  - UP/TP:   2.5426 / 2.5474 = 0.998x uniqueness")

print("\n⚠️  PROBLEM DETECTED:")
print("Your uniqueness multipliers are nearly 1.0 for all classes!")
print("This means uniqueness weights are ~uniform (not down-weighting overlaps)")
print("\nExpected if LdP working properly:")
print("  - High overlap (TIMEOUT): uniqueness ~0.3-0.5x (down-weighted)")
print("  - Low overlap (barrier hits): uniqueness ~1.5-3.0x (up-weighted)")
print("\nYour actual:")
print("  - All classes: uniqueness ~0.98-1.03x (essentially uniform)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("✅ The weights you reported ARE essentially the same as sklearn!")
print("❌ López de Prado uniqueness weighting is NOT working as intended")
print("\nPossible causes:")
print("1. All labels have similar overlap (unlikely with 65% TIMEOUT)")
print("2. Implementation bug in averaging uniqueness")
print("3. Events dataframe not properly aligned with sequences")

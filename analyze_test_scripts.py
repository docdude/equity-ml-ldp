"""
Analyze all test scripts to identify:
1. Which ones load data directly (need NaN fix)
2. Which ones use load_or_generate_sequences (already protected)
3. What validation methods they use
"""

import os
import re
from pathlib import Path

# Find all test scripts
test_files = sorted(Path('.').glob('test_*.py'))

print("="*80)
print("TEST SCRIPT ANALYSIS - NaN BUG VULNERABILITY")
print("="*80)

results = []

for test_file in test_files:
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Check for direct data loading
    has_read_parquet = 'read_parquet' in content or 'read_csv' in content
    uses_load_sequences = 'load_or_generate_sequences' in content
    has_dropna_ohlcv = bool(re.search(r"dropna\(subset=\[.*open.*high.*low.*close.*volume", content))
    uses_enhanced_features = 'EnhancedFinancialFeatures' in content
    calls_create_all_features = 'create_all_features' in content
    
    # Extract validation methods
    validation_methods = []
    if 'talib' in content.lower():
        validation_methods.append('TA-Lib comparison')
    if 'allclose' in content or 'isclose' in content:
        validation_methods.append('Numeric comparison')
    if 'range' in content.lower() and ('min' in content or 'max' in content):
        validation_methods.append('Range validation')
    if 'nan' in content.lower():
        validation_methods.append('NaN checks')
    
    # Determine vulnerability
    if uses_load_sequences:
        status = "✅ PROTECTED (uses fixed function)"
    elif has_dropna_ohlcv:
        status = "✅ PROTECTED (has OHLCV dropna)"
    elif has_read_parquet and uses_enhanced_features:
        status = "❌ VULNERABLE (direct load + feature gen)"
    elif has_read_parquet:
        status = "⚠️  MAYBE (loads data, check manually)"
    else:
        status = "ℹ️  N/A (no data loading)"
    
    results.append({
        'file': test_file.name,
        'status': status,
        'direct_load': has_read_parquet,
        'uses_sequences': uses_load_sequences,
        'has_dropna': has_dropna_ohlcv,
        'uses_features': uses_enhanced_features,
        'methods': validation_methods
    })

# Print results by category
print("\n📊 SUMMARY BY VULNERABILITY STATUS\n")

for status_type in ["❌ VULNERABLE", "⚠️  MAYBE", "✅ PROTECTED", "ℹ️  N/A"]:
    matching = [r for r in results if r['status'].startswith(status_type[:3])]
    if matching:
        print(f"\n{status_type} ({len(matching)} files)")
        print("-"*80)
        for r in matching:
            print(f"\n  📄 {r['file']}")
            print(f"     Direct load: {r['direct_load']}")
            print(f"     Uses sequences: {r['uses_sequences']}")
            print(f"     Has OHLCV dropna: {r['has_dropna']}")
            print(f"     Creates features: {r['uses_features']}")
            if r['methods']:
                print(f"     Validation: {', '.join(r['methods'])}")

# Detailed recommendations
print("\n" + "="*80)
print("DETAILED RECOMMENDATIONS")
print("="*80)

vulnerable = [r for r in results if r['status'].startswith("❌")]
if vulnerable:
    print(f"\n🔧 FIX NEEDED ({len(vulnerable)} files):")
    for r in vulnerable:
        print(f"\n  {r['file']}")
        print(f"  → Add after read_parquet:")
        print(f"     df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])")

maybe = [r for r in results if r['status'].startswith("⚠️")]
if maybe:
    print(f"\n🔍 REVIEW NEEDED ({len(maybe)} files):")
    for r in maybe:
        print(f"  - {r['file']}")

protected = [r for r in results if r['status'].startswith("✅")]
print(f"\n✅ ALREADY PROTECTED ({len(protected)} files)")

print("\n" + "="*80)
print("VALIDATION METHODS INVENTORY")
print("="*80)

all_methods = set()
for r in results:
    all_methods.update(r['methods'])

print("\nValidation methods found across all test scripts:")
for method in sorted(all_methods):
    files_with_method = [r['file'] for r in results if method in r['methods']]
    print(f"\n  {method} ({len(files_with_method)} files):")
    for f in files_with_method[:5]:  # Show first 5
        print(f"    - {f}")
    if len(files_with_method) > 5:
        print(f"    ... and {len(files_with_method)-5} more")


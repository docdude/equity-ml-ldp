#!/usr/bin/env python3
"""
Verify all test scripts have been fixed for the NaN bug
Run this to check if fixes have been applied
"""

import re
from pathlib import Path

print("="*80)
print("VERIFICATION: NaN BUG FIXES IN TEST SCRIPTS")
print("="*80)

# Test files that were identified as vulnerable
vulnerable_files = [
    'test_all_features.py',
    'test_dataset_validation.py',
    'test_feature_accuracy.py',
    'test_feature_accuracy_simple.py',
    'test_feature_ranges.py',
    'test_features.py',
    'test_full_features.py',
    'test_normalization_effect.py',
    'test_pbo_quick.py'
]

fixed = []
not_fixed = []

for filename in vulnerable_files:
    filepath = Path(filename)
    if not filepath.exists():
        print(f"⚠️  {filename} - FILE NOT FOUND")
        continue
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for the fix pattern (now using capitalized column names)
    has_read_parquet = 'read_parquet' in content
    has_dropna_fix = bool(re.search(
        r"dropna\(subset=\[.*['\"]Open['\"].*['\"]High['\"].*['\"]Low['\"].*['\"]Close['\"].*['\"]Volume['\"].*\]",
        content
    ))
    
    if has_read_parquet and has_dropna_fix:
        fixed.append(filename)
        print(f"✅ {filename} - FIXED")
    elif has_read_parquet:
        not_fixed.append(filename)
        print(f"❌ {filename} - NOT FIXED (loads data but missing dropna)")
    else:
        print(f"ℹ️  {filename} - NO DATA LOADING")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal vulnerable files: {len(vulnerable_files)}")
print(f"Fixed: {len(fixed)}")
print(f"Not fixed: {len(not_fixed)}")

if not_fixed:
    print(f"\n⚠️  Files still need fixing:")
    for f in not_fixed:
        print(f"  - {f}")
else:
    print(f"\n🎉 All vulnerable test scripts have been fixed!")

print(f"\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if not_fixed:
    print("\n1. Fix remaining test scripts")
    print("2. Run comprehensive validation")
else:
    print("\n1. ✅ All test scripts fixed")
    print("2. Run comprehensive validation:")
    print("   .venv/bin/python test_all_features.py")
    print("   .venv/bin/python test_feature_accuracy.py")
    print("   .venv/bin/python test_full_features.py")
    print("\n3. Verify features pass validation:")
    print("   .venv/bin/python validate_wavenet_v2_features.py")

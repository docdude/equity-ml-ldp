"""Quick test to verify wavenet_optimized feature selection"""

from feature_config import FeatureConfig

# Test 1: Load wavenet_optimized preset
print("="*60)
print("TEST: wavenet_optimized Preset")
print("="*60)

config = FeatureConfig.get_preset('wavenet_optimized')

print("\n1. Config keys:")
for key in config.keys():
    print(f"   - {key}")

print("\n2. Feature groups enabled:")
feature_groups = {k: v for k, v in config.items() if isinstance(v, bool)}
for group, enabled in feature_groups.items():
    status = "✅" if enabled else "❌"
    print(f"   {status} {group}: {enabled}")

print("\n3. Feature list:")
if 'feature_list' in config:
    feature_list = config['feature_list']
    print(f"   Total features: {len(feature_list)}")
    for i, feat in enumerate(feature_list, 1):
        print(f"   {i:2d}. {feat}")
else:
    print("   ⚠️  WARNING: No 'feature_list' found in config!")

print("\n4. Expected features: 18")
print(f"   Actual features: {len(config.get('feature_list', []))}")
if len(config.get('feature_list', [])) == 18:
    print("   ✅ PASS: Feature count matches!")
else:
    print("   ❌ FAIL: Feature count mismatch!")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)

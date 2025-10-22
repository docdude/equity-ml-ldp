"""
Compare the latest feature importance analysis with wavenet_optimized_v2 config
"""

import pandas as pd
from feature_config import FeatureConfig

print("\n" + "="*80)
print("FEATURE RANKING COMPARISON")
print("="*80)

# Load the latest analysis results
mdi_df = pd.read_csv('artifacts/feature_importance/MDI_importance.csv', index_col=0)
mda_df = pd.read_csv('artifacts/feature_importance/MDA_importance.csv', index_col=0)
ortho_mda_df = pd.read_csv('artifacts/feature_importance/Ortho_MDA_importance_mapped.csv')

# Get current wavenet_optimized_v2 config
config = FeatureConfig.get_preset('wavenet_optimized_v2')
current_features = set(config['feature_list'])

print(f"\nCurrent wavenet_optimized_v2 has {len(current_features)} features\n")

# Create rankings
mdi_df['mdi_rank'] = range(1, len(mdi_df) + 1)
mda_df['mda_rank'] = range(1, len(mda_df) + 1)
ortho_mda_df['ortho_rank'] = range(1, len(ortho_mda_df) + 1)

# Get top 40 from each method
top_n = 40
mdi_top = set(mdi_df.head(top_n).index)
mda_top = set(mda_df.head(top_n).index)
ortho_top = set(ortho_mda_df.head(top_n)['feature'].values)

print("="*80)
print("TOP 40 FEATURES FROM EACH METHOD")
print("="*80)

print("\n📊 MDI Top 40:")
print(", ".join(sorted(list(mdi_top)[:15])))
print("   ...")

print("\n📊 MDA Top 40:")
print(", ".join(sorted(list(mda_top)[:15])))
print("   ...")

print("\n📊 Ortho MDA Top 40:")
print(", ".join(sorted(list(ortho_top)[:15])))
print("   ...")

# Calculate consensus features (appear in at least 2 of 3 methods)
all_features = mdi_top | mda_top | ortho_top
consensus_features = []

for feature in all_features:
    count = 0
    ranks = []
    
    if feature in mdi_top:
        count += 1
        ranks.append(('MDI', mdi_df.loc[feature, 'mdi_rank']))
    if feature in mda_top:
        count += 1
        ranks.append(('MDA', mda_df.loc[feature, 'mda_rank']))
    if feature in ortho_top:
        count += 1
        ortho_rank = ortho_mda_df[ortho_mda_df['feature'] == feature]['ortho_rank'].values[0]
        ranks.append(('Ortho', ortho_rank))
    
    if count >= 2:  # Consensus: appears in at least 2 methods
        avg_rank = sum(r for _, r in ranks) / len(ranks)
        consensus_features.append({
            'feature': feature,
            'consensus_count': count,
            'avg_rank': avg_rank,
            'ranks': ranks
        })

# Sort by consensus count (desc), then avg rank (asc)
consensus_features.sort(key=lambda x: (-x['consensus_count'], x['avg_rank']))

print("\n" + "="*80)
print(f"CONSENSUS FEATURES (appear in 2+ methods, top {top_n})")
print("="*80)

print(f"\nTotal consensus features: {len(consensus_features)}")
print(f"  - In all 3 methods: {sum(1 for f in consensus_features if f['consensus_count'] == 3)}")
print(f"  - In 2 methods: {sum(1 for f in consensus_features if f['consensus_count'] == 2)}")

# Get top 40 consensus
consensus_top40 = [f['feature'] for f in consensus_features[:40]]

print("\n📋 Top 40 Consensus Features:")
for i, feat_info in enumerate(consensus_features[:40], 1):
    feat = feat_info['feature']
    count = feat_info['consensus_count']
    avg_rank = feat_info['avg_rank']
    ranks_str = ", ".join([f"{method}:{int(rank)}" for method, rank in feat_info['ranks']])
    in_current = "✅" if feat in current_features else "❌"
    print(f"  {i:2d}. {feat:30s} [{count}/3] avg_rank:{avg_rank:5.1f} ({ranks_str}) {in_current}")

# Compare with current config
print("\n" + "="*80)
print("COMPARISON WITH CURRENT WAVENET_OPTIMIZED_V2")
print("="*80)

new_consensus = set(consensus_top40)
in_both = current_features & new_consensus
only_old = current_features - new_consensus
only_new = new_consensus - current_features

print(f"\n✅ Features in BOTH (current config AND new consensus): {len(in_both)}")
for feat in sorted(in_both):
    print(f"   • {feat}")

print(f"\n❌ Features ONLY in current config (not in new top 40): {len(only_old)}")
for feat in sorted(only_old):
    # Find rank in new analysis
    ranks = []
    if feat in mdi_df.index:
        ranks.append(f"MDI:{int(mdi_df.loc[feat, 'mdi_rank'])}")
    if feat in mda_df.index:
        ranks.append(f"MDA:{int(mda_df.loc[feat, 'mda_rank'])}")
    if feat in ortho_mda_df['feature'].values:
        ortho_rank = ortho_mda_df[ortho_mda_df['feature'] == feat]['ortho_rank'].values[0]
        ranks.append(f"Ortho:{int(ortho_rank)}")
    ranks_str = ", ".join(ranks) if ranks else "N/A"
    print(f"   • {feat:30s} [{ranks_str}]")

print(f"\n🆕 Features ONLY in new consensus (not in current config): {len(only_new)}")
for feat in sorted(only_new):
    # Find consensus info
    feat_info = next((f for f in consensus_features if f['feature'] == feat), None)
    if feat_info:
        count = feat_info['consensus_count']
        avg_rank = feat_info['avg_rank']
        ranks_str = ", ".join([f"{method}:{int(rank)}" for method, rank in feat_info['ranks']])
        print(f"   • {feat:30s} [{count}/3] avg_rank:{avg_rank:5.1f} ({ranks_str})")

# Filter out market features
market_features = {'GC=F', '^VIX', '^GSPC', '^TYX', '^FVX', 'JPY=X'}
consensus_top40_no_market = [f for f in consensus_top40 if f not in market_features]

print("\n" + "="*80)
print("RECOMMENDED FEATURES (excluding market features)")
print("="*80)
print(f"\nTop 40 consensus features (excluding {len([f for f in consensus_top40 if f in market_features])} market features):")
print(f"Total: {len(consensus_top40_no_market)} features\n")

for i, feat in enumerate(consensus_top40_no_market[:40], 1):
    in_current = "✅" if feat in current_features else "🆕"
    print(f"  {i:2d}. {feat:30s} {in_current}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Current wavenet_optimized_v2: {len(current_features)} features
New consensus top 40: {len(consensus_top40_no_market)} features (excluding market)

Overlap: {len(current_features & set(consensus_top40_no_market))} features
To remove: {len(current_features - set(consensus_top40_no_market))} features
To add: {len(set(consensus_top40_no_market) - current_features)} features

Recommendation: {'✅ Current config is well-aligned' if len(current_features & set(consensus_top40_no_market)) >= 30 else '⚠️ Consider updating config with new consensus features'}
""")

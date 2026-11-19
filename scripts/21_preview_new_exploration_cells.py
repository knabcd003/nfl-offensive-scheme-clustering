import pandas as pd
import numpy as np

# Load clean team-season features
# In the notebook, after Cell 17, we have df_clean containing FEATS and Z_FEATS
df_pbp = pd.read_parquet('data_cache/pbp_neutral_clean.parquet')
DEEP_YARDS = 15
records = []
for (season, team), grp in df_pbp.groupby(['season', 'posteam']):
    n = len(grp)
    if n < 50:
        continue
    passes = grp[grp['play_type'] == 'pass']
    runs   = grp[grp['play_type'] == 'run']
    ay     = passes['air_yards'].dropna()
    loc_dist = passes['pass_location'].value_counts(normalize=True)
    interior = runs['run_location'].isin(['middle', 'left guard', 'right guard'])
    records.append({
        'season': int(season),
        'team': team,
        'pass_rate': grp['qb_dropback'].fillna(0).mean(),
        'shotgun_rate': grp['shotgun'].mean(),
        'no_huddle_rate': grp['no_huddle'].mean(),
        'air_yards_per_att': ay.mean() if len(ay) else np.nan,
        'deep_target_rate': (ay >= DEEP_YARDS).mean() if len(ay) else np.nan,
        'positional_target_conc': (loc_dist**2).sum() if len(loc_dist) else np.nan,
        'rush_direction_conc': interior.mean() if len(runs) else np.nan,
    })

df_raw = pd.DataFrame(records)
FEATS = ['pass_rate', 'shotgun_rate', 'no_huddle_rate', 'air_yards_per_att',
         'deep_target_rate', 'positional_target_conc', 'rush_direction_conc']
Z_FEATS = [f'{f}_z' for f in FEATS]

df_z = df_raw.copy()
for f in FEATS:
    mu  = df_z.groupby('season')[f].transform('mean')
    sig = df_z.groupby('season')[f].transform('std')
    df_z[f'{f}_z'] = (df_z[f] - mu) / sig

df_clean = df_z.dropna(subset=Z_FEATS).reset_index(drop=True)

# 1. Summary stats table
stats_table = pd.DataFrame({
    'Mean': df_clean[FEATS].mean(),
    'Std': df_clean[FEATS].std(),
    'Median': df_clean[FEATS].median(),
    'Min': df_clean[FEATS].min(),
    'Max': df_clean[FEATS].max(),
    'Skewness': df_clean[FEATS].skew()
})
print("Summary Statistics Table:")
print(stats_table.round(3))

# 2. Extreme Outliers Table (|Z| > 3.0)
outliers = []
for f, zf in zip(FEATS, Z_FEATS):
    ext = df_clean[df_clean[zf].abs() > 3.0][['season', 'team', f, zf]]
    for _, r in ext.iterrows():
        outliers.append({
            'Season': int(r['season']),
            'Team': r['team'],
            'Feature': f,
            'Raw Value': r[f],
            'Z-Score (σ)': r[zf]
        })
out_df = pd.DataFrame(outliers).sort_values('Z-Score (σ)', key=abs, ascending=False)
print(f"\nTop 10 Extreme Tactical Outliers (|Z| > 3.0):")
print(out_df.head(10).round(2).to_string(index=False))

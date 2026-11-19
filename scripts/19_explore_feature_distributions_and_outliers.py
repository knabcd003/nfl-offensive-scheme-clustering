import pandas as pd
import numpy as np

# Load the clean dataset or run the feature pipeline
df_pbp = pd.read_parquet('data_cache/pbp_neutral_clean.parquet')
print(f"PBP loaded: {len(df_pbp):,} plays")

# Build tactical features
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
        'n_plays': n,
        'pass_rate': grp['qb_dropback'].fillna(0).mean(),
        'shotgun_rate': grp['shotgun'].mean(),
        'no_huddle_rate': grp['no_huddle'].mean(),
        'air_yards_per_att': ay.mean() if len(ay) else np.nan,
        'deep_target_rate': (ay >= DEEP_YARDS).mean() if len(ay) else np.nan,
        'positional_target_conc': (loc_dist**2).sum() if len(loc_dist) else np.nan,
        'rush_direction_conc': interior.mean() if len(runs) else np.nan,
    })

df_raw = pd.DataFrame(records)

# Within-season Z-scores
FEATS = ['pass_rate', 'shotgun_rate', 'no_huddle_rate', 'air_yards_per_att',
         'deep_target_rate', 'positional_target_conc', 'rush_direction_conc']
Z_FEATS = [f'{f}_z' for f in FEATS]

df_z = df_raw.copy()
for f in FEATS:
    mu  = df_z.groupby('season')[f].transform('mean')
    sig = df_z.groupby('season')[f].transform('std')
    df_z[f'{f}_z'] = (df_z[f] - mu) / sig

# Filter to 2006-2024
df_clean = df_z.dropna(subset=Z_FEATS).reset_index(drop=True)
print(f"Clean team-seasons (2006-2024): {len(df_clean)}")

# 1. Summary statistics of raw features
print("\n--- Summary Statistics of Raw Tactical Features (2006-2024) ---")
summary_raw = df_clean[FEATS].describe().T[['mean', 'std', 'min', '50%', 'max']]
summary_raw['skew'] = df_clean[FEATS].skew()
print(summary_raw.round(3))

# 2. Outlier Analysis (|Z| > 3.0)
print("\n--- Outliers with |Z| > 3.0 (Extreme Tactical Outliers) ---")
outlier_list = []
for f, zf in zip(FEATS, Z_FEATS):
    extreme = df_clean[df_clean[zf].abs() > 3.0][['season', 'team', f, zf]]
    for _, row in extreme.iterrows():
        outlier_list.append({
            'season': int(row['season']),
            'team': row['team'],
            'feature': f,
            'raw_value': row[f],
            'z_score': row[zf]
        })

outliers_df = pd.DataFrame(outlier_list)
if not outliers_df.empty:
    print(outliers_df.sort_values('z_score', ascending=False).to_string(index=False))
else:
    print("No outliers with |Z| > 3.0")

# Check outliers with |Z| > 2.5
print(f"\nTotal observations with any feature |Z| > 2.5: {df_clean[(df_clean[Z_FEATS].abs() > 2.5).any(axis=1)].shape[0]}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['figure.dpi'] = 110

# Load from clean script
df_clean = pd.read_parquet('data_cache/df_clean_test.parquet') if False else None

# Run feature computation from pbp
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

# Create 7-feature distribution plot
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

feature_titles = [
    'Dropback Pass Rate', 'Shotgun Snap Rate', 'No-Huddle Tempo Rate',
    'Intended Air Yards / Att', 'Deep Target Rate (≥15 yds)',
    'Target Dispersion (HHI)', 'Interior Rush Share'
]

for i, (f, title) in enumerate(zip(FEATS, feature_titles)):
    sns.histplot(df_clean[f], kde=True, ax=axes[i], color='steelblue', bins=25)
    axes[i].set_title(title, fontweight='bold', fontsize=11)
    axes[i].set_xlabel('Raw Proportion / Value')
    axes[i].set_ylabel('Team-Seasons')

# Remove 8th empty subplot
fig.delaxes(axes[7])
plt.suptitle('Distributions of 7 Raw Tactical Features (608 Team-Seasons, 2006–2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('scripts/feature_distributions.png')
print("Distributions plot saved.")

# Standardized Z-Score Boxplots
plt.figure(figsize=(12, 4.5))
short_names = ['Pass Rate', 'Shotgun', 'No-Huddle', 'Air Yards', 'Deep Tgt %', 'Target Conc', 'Interior Rush']
sns.boxplot(data=df_clean[Z_FEATS], palette='Set2')
plt.xticks(range(7), short_names, fontweight='bold')
plt.ylabel('Within-Season Z-Score (σ)')
plt.title('Standardized Tactical Feature Spread and Outliers', fontweight='bold')
plt.axhline(0, color='gray', linestyle='--', lw=0.8)
plt.axhline(3.0, color='red', linestyle=':', lw=0.8, label='|Z| = 3.0 threshold')
plt.axhline(-3.0, color='red', linestyle=':', lw=0.8)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('scripts/feature_boxplots.png')
print("Boxplots saved.")


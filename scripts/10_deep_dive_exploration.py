import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Let's inspect the cached pbp neutral data
cache_dir = Path("data_cache")
pbp_cache = cache_dir / "pbp_neutral_clean.parquet"
perf_cache = cache_dir / "team_season_perf.parquet"

print("Checking files...")
print("PBP exists:", pbp_cache.exists())
print("Perf exists:", perf_cache.exists())

df_pbp = pd.read_parquet(pbp_cache)
print(f"PBP shape: {df_pbp.shape}")
print(f"PBP columns ({len(df_pbp.columns)}):", df_pbp.columns.tolist()[:30])

# Inspect seasons and play counts
print("\nPlays per season:")
print(df_pbp['season'].value_counts().sort_index())

# Check columns related to pass/rush locations and targets
target_cols = [c for c in df_pbp.columns if any(k in c for k in ['location', 'target', 'receiver', 'pos', 'passer', 'rusher', 'down'])]
print("\nRelevant target/location/down columns in pbp:", target_cols)

# Check missingness of key columns across seasons
cols_to_check = [c for c in ['down', 'air_yards', 'pass_location', 'run_location', 'shotgun', 'no_huddle', 'qb_dropback'] if c in df_pbp.columns]
missing_by_season = df_pbp.groupby('season')[cols_to_check].apply(lambda g: g.isna().mean())
print("\nMissing rate by season (sample):")
print(missing_by_season.loc[[2002, 2003, 2004, 2005, 2006, 2010, 2015, 2020, 2024]])


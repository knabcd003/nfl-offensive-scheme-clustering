import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.stattools import durbin_watson
from scipy.cluster.hierarchy import linkage

# Set MPLCONFIGDIR
import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'

# Verify all data files and transforms
cache_dir = Path("data_cache")
df_pbp = pd.read_parquet(cache_dir / "pbp_neutral_clean.parquet")
df_perf = pd.read_parquet(cache_dir / "team_season_perf.parquet")

print("PBP loaded:", len(df_pbp))
print("Perf loaded:", len(df_perf))

# Test primary QB logic
passer_counts = (df_pbp[df_pbp['play_type'] == 'pass']
                 .groupby(['season', 'posteam', 'passer_player_name'])
                 .size()
                 .reset_index(name='pass_attempts'))
idx_max = passer_counts.groupby(['season', 'posteam'])['pass_attempts'].idxmax()
primary_passers = passer_counts.loc[idx_max].rename(columns={'posteam': 'team', 'passer_player_name': 'primary_qb'})
primary_passers['team_clean'] = primary_passers['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
primary_passers = primary_passers.sort_values(['team_clean', 'season'])
primary_passers['next_primary_qb'] = primary_passers.groupby('team_clean')['primary_qb'].shift(-1)
primary_passers['qb_continuity'] = (primary_passers['primary_qb'] == primary_passers['next_primary_qb']).astype(int)

print("Primary QB count:", len(primary_passers))
print("Flow test completed successfully.")

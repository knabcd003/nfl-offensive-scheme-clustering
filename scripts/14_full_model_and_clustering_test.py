import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.stattools import durbin_watson

# 1. Load data
df_z = pd.read_parquet('/Users/karthikn/.gemini/antigravity/brain/0a7858a8-4baa-4df7-af5e-b6156637325d/scratch/df_z_clean.parquet')
df_perf = pd.read_parquet('/Users/karthikn/.gemini/antigravity/brain/0a7858a8-4baa-4df7-af5e-b6156637325d/scratch/df_perf.parquet')
df_pbp = pd.read_parquet("data_cache/pbp_neutral_clean.parquet")

FEATS = ['pass_rate', 'shotgun_rate', 'no_huddle_rate', 'air_yards_per_att',
         'deep_target_rate', 'positional_target_conc', 'rush_direction_conc']
Z_FEATS = [f'{f}_z' for f in FEATS]

# Only complete seasons (2006-2024)
df = df_z.dropna(subset=Z_FEATS).copy().reset_index(drop=True)
X = df[Z_FEATS].values

print(f"Data shape: {df.shape}, X shape: {X.shape}")

# 2. Clustering algorithm comparison across k=2..6
records = []
for k in range(2, 7):
    # KMeans
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42)
    km_labels = km.fit_predict(X)
    
    # Agglomerative (Ward)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labels = agg.fit_predict(X)
    
    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    
    records.append({
        'k': k,
        'km_sil': silhouette_score(X, km_labels),
        'agg_sil': silhouette_score(X, agg_labels),
        'gmm_sil': silhouette_score(X, gmm_labels),
        'km_db': davies_bouldin_score(X, km_labels),
        'agg_db': davies_bouldin_score(X, agg_labels),
        'gmm_db': davies_bouldin_score(X, gmm_labels),
        'gmm_bic': gmm.bic(X),
        'km_inertia': km.inertia_
    })

eval_df = pd.DataFrame(records)
print("\nClustering metric comparison across algorithms:")
print(eval_df.round(3))

# Final k=3 KMeans
km3 = KMeans(n_clusters=3, init='k-means++', n_init=100, random_state=42)
df['cluster'] = km3.fit_predict(X)

# Compute Euclidean distance to each centroid for continuous representation
for c in range(3):
    df[f'dist_centroid_{c}'] = np.linalg.norm(X - km3.cluster_centers_[c], axis=1)

print("\nCluster counts:")
print(df['cluster'].value_counts())

# 3. Add QB continuity
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

# Merge performance and QB continuity
df['team_clean'] = df['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
df_perf['team_clean'] = df_perf['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})

df_merged = df.merge(df_perf[['season', 'team_clean', 'net_pts_per_game']], on=['season', 'team_clean'], how='left')
df_merged = df_merged.merge(primary_passers[['season', 'team_clean', 'qb_continuity', 'primary_qb']], on=['season', 'team_clean'], how='left')

# Create target y_{t+1}
df_merged = df_merged.sort_values(['team_clean', 'season'])
df_merged['y_next'] = df_merged.groupby('team_clean')['net_pts_per_game'].shift(-1)
df_model = df_merged.dropna(subset=['y_next', 'net_pts_per_game', 'qb_continuity']).reset_index(drop=True)
print(f"\nFinal modeling sample size: {len(df_model)} team-seasons")

# 4. Supervised CV Benchmark
CV_FOLDS = [
    {'train_end': 2010, 'eval_start': 2011, 'eval_end': 2013},
    {'train_end': 2013, 'eval_start': 2014, 'eval_end': 2016},
    {'train_end': 2016, 'eval_start': 2017, 'eval_end': 2019},
    {'train_end': 2019, 'eval_start': 2020, 'eval_end': 2024},
]

feature_sets = {
    '1. Baseline (Y_t)': ['net_pts_per_game'],
    '2. Baseline + QB Continuity': ['net_pts_per_game', 'qb_continuity'],
    '3. Baseline + Style Distances': ['net_pts_per_game', 'dist_centroid_0', 'dist_centroid_1', 'dist_centroid_2'],
    '4. Full Model (Y_t + QB + Style)': ['net_pts_per_game', 'qb_continuity', 'dist_centroid_0', 'dist_centroid_1', 'dist_centroid_2']
}

cv_results = []
for f_name, f_cols in feature_sets.items():
    for fold in CV_FOLDS:
        tr = df_model[df_model['season'] <= fold['train_end']]
        te = df_model[(df_model['season'] >= fold['eval_start']) & (df_model['season'] <= fold['eval_end'])]
        Xtr, ytr = tr[f_cols].values, tr['y_next'].values
        Xte, yte = te[f_cols].values, te['y_next'].values
        
        models = [
            ('Ridge', Ridge(alpha=10.0)),
            ('Random Forest', RandomForestRegressor(max_depth=3, min_samples_leaf=15, n_estimators=100, random_state=42)),
            ('LightGBM', lgb.LGBMRegressor(learning_rate=0.03, num_leaves=6, reg_lambda=15.0, n_estimators=100, random_state=42, verbose=-1))
        ]
        
        for m_name, model in models:
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            cv_results.append({
                'Feature Set': f_name,
                'Model': m_name,
                'Fold': f"{fold['eval_start']}-{fold['eval_end']}",
                'RMSE': np.sqrt(mean_squared_error(yte, preds)),
                'MAE': mean_absolute_error(yte, preds),
                'R2': r2_score(yte, preds)
            })

df_cv = pd.DataFrame(cv_results)
summary = df_cv.groupby(['Feature Set', 'Model'])[['RMSE', 'MAE', 'R2']].mean().round(3)
print("\nCross-Validation Summary (Out-of-sample averages):")
print(summary)

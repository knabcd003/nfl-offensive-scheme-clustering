import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

df_z = pd.read_parquet('/Users/karthikn/.gemini/antigravity/brain/0a7858a8-4baa-4df7-af5e-b6156637325d/scratch/df_z_clean.parquet')
FEATS = ['pass_rate', 'shotgun_rate', 'no_huddle_rate', 'air_yards_per_att',
         'deep_target_rate', 'positional_target_conc', 'rush_direction_conc']
Z_FEATS = [f'{f}_z' for f in FEATS]
df = df_z.dropna(subset=Z_FEATS).copy().reset_index(drop=True)
X = df[Z_FEATS].values

Z_ward = linkage(X, method='ward')
print("Ward linkage shape:", Z_ward.shape)
print("Top 5 merges in Ward linkage (distances):", Z_ward[-5:, 2])


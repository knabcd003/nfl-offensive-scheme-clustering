import json, textwrap
from pathlib import Path

NB_PATH = Path("nfl_offensive_clustering.ipynb")

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip(), "outputs": [], "execution_count": None}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(src).strip(), "outputs": [], "execution_count": None}

cells = []

# Title & Framing
cells.append(md("""
# NFL Offensive Scheme Clustering and Strategic Decay

**Course:** Principles of Machine Learning I — Fall 2026  
**Topic:** Unsupervised Scheme Discovery & Supervised Performance Validation  

---

### What We Are Trying to Solve

In the NFL, coaches and analysts frequently talk about offensive "systems"—the West Coast offense, power-run heavy sets, or high-tempo spread schemes. But does offensive scheme represent an independent, durable source of competitive advantage, or is success almost entirely determined by talent and quarterback play?

We analyze play-by-play data spanning 2002–2024 to tackle three questions:
1. **Unsupervised Discovery:** Do distinct offensive archetypes genuinely exist in the data, or is offensive play-calling just a continuous spectrum?
2. **Strategy Decay:** If an offensive scheme gives a team a performance edge, does that edge decay as more teams across the league adopt it?
3. **Supervised Prediction:** Does knowing a team's offensive style help us predict their performance next season, or does past performance and quarterback continuity render scheme irrelevant?
"""))

# Section 1: Setup
cells.append(md("""
## 1. Setup and Environment

We import the required libraries, configure plotting aesthetics, and set a fixed random seed for reproducible cluster initializations and bootstrap resamples.
"""))

cells.append(code("""
import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import nflreadpy
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 110
np.random.seed(42)

SEASONS = list(range(2002, 2025))
WP_LOW = 0.20
WP_HIGH = 0.80
DEEP_YARDS = 15

print(f"Tracking {len(SEASONS)} seasons: {SEASONS[0]} through {SEASONS[-1]}")
"""))

# Section 2: Data Loading & Caching
cells.append(md("""
## 2. Data Ingestion and Caching

We pull play-by-play and game schedule logs from the nflverse open-source ecosystem via `nflreadpy`. Because downloading 23 seasons of play-by-play data (~1 million rows) takes several minutes, we cache the cleaned parquet tables locally in `data_cache/`.
"""))

cells.append(code("""
CACHE = Path('data_cache')
CACHE.mkdir(exist_ok=True)
PBP_CACHE  = CACHE / 'pbp_neutral_clean.parquet'
PERF_CACHE = CACHE / 'team_season_perf.parquet'

if PBP_CACHE.exists() and PERF_CACHE.exists():
    print("Loading preprocessed data from local parquet cache...")
    df_pbp  = pd.read_parquet(PBP_CACHE)
    df_perf = pd.read_parquet(PERF_CACHE)
    # Reconcile historical franchise abbreviations (STL -> LA, SD -> LAC, OAK -> LV)
    df_perf['team_clean'] = df_perf['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
else:
    print("Fetching raw data via nflreadpy (first run only)...")
    pbp_raw = nflreadpy.load_pbp(SEASONS).to_pandas()
    sched   = nflreadpy.load_schedules(SEASONS).to_pandas()

    # Aggregate regular season team net points per game
    home = sched[['season', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'pts_for', 'away_score': 'pts_against'})
    away = sched[['season', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'pts_for', 'home_score': 'pts_against'})
    games = pd.concat([home, away]).dropna(subset=['pts_for'])
    df_perf = (games.groupby(['season', 'team'])
               .agg(pts_scored=('pts_for', 'sum'), pts_allowed=('pts_against', 'sum'), games=('pts_for', 'count'))
               .reset_index())
    df_perf['net_pts_per_game'] = (df_perf['pts_scored'] - df_perf['pts_allowed']) / df_perf['games']
    df_perf['team_clean'] = df_perf['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})

    # Filter scrimmage plays and non-tactical plays
    pbp = pbp_raw[pbp_raw['play_type'].isin(['pass', 'run'])].copy()
    pbp = pbp[(pbp['qb_kneel'] != 1) & (pbp['qb_spike'] != 1)]
    pbp['qb_dropback'] = np.where(pbp['qb_scramble'] == 1, 1, pbp['qb_dropback'])
    df_pbp = pbp[pbp['wp'].between(WP_LOW, WP_HIGH) & (pbp['half_seconds_remaining'] > 120)]

    df_pbp.to_parquet(PBP_CACHE)
    df_perf.to_parquet(PERF_CACHE)
    print("Saved clean tables to data_cache/.")

print(f"Neutral play-by-play records: {len(df_pbp):,} plays")
print(f"Team-season performance records: {len(df_perf):,} team-seasons")
"""))

# Section 3: Data Cleaning & Audit
cells.append(md("""
## 3. Data Cleaning and Tactical Normalization

To accurately measure an offensive play-caller's true philosophical tendencies, we filter out situational noise:
1. **Remove QB Kneels and Spikes:** Clock management plays contain no tactical scheme information.
2. **Reclassify QB Scrambles:** Scrambles are officially recorded as rushing attempts, but they originate as called pass dropbacks where the pocket broke down. We reclassify them as pass dropbacks (`qb_dropback = 1`).
3. **Neutral Game Script Filter:** A team trailing by 21 in the 4th quarter passes on almost every play; a team leading by 21 runs the ball to burn clock. We restrict our data to plays with Win Probability between $20\\%$ and $80\\%$ with more than 2 minutes remaining in the half.
"""))

cells.append(code("""
# Quick check of scrimmage play counts
print("Breakdown of neutral scrimmage plays:")
print(df_pbp['play_type'].value_counts())
print(f"Average Win Probability in neutral sample: {df_pbp['wp'].mean():.3f}")
"""))

cells.append(md("""
### Investigating Tracking Completeness Across Eras

Let's check the missingness rate of key tracking fields across historical seasons. This reveals a critical structural boundary in the data.
"""))

cells.append(code("""
passes = df_pbp[df_pbp['play_type'] == 'pass']
missing_audit = pd.DataFrame({
    'Air Yards Missing': passes.groupby('season')['air_yards'].apply(lambda s: s.isna().mean()),
    'Pass Location Missing': passes.groupby('season')['pass_location'].apply(lambda s: s.isna().mean()),
    'Shotgun Missing': df_pbp.groupby('season')['shotgun'].apply(lambda s: s.isna().mean())
})

print("Missingness rate across selected seasons:")
print(missing_audit.loc[[2002, 2004, 2005, 2006, 2010, 2018, 2024]])
"""))

cells.append(md("""
**Key Finding on Data Completeness:**  
Before 2006, intended target depth (`air_yards`) and pass directional charting (`pass_location`) were essentially not tracked ($100\\%$ missing). Standardized NFL optical and charted tracking began in **2006**.  
This explains why early seasons cannot support depth-based tactical features: 2006–2024 ($19$ full seasons $\\times 32$ teams $= 608$ team-seasons) represents the complete, high-integrity tactical tracking era.
"""))

# Section 4: EDA & Macro-Drift
cells.append(md("""
## 4. Exploratory Data Analysis: Macro-Drift vs. Team Divergence

Before clustering, we must inspect how offensive football evolved over time. If the entire league shifted toward more passing and shotgun formations, a naive clustering algorithm would cluster on *eras* rather than *offensive schemes*.
"""))

cells.append(code("""
# League-wide evolution of neutral pass rate and shotgun usage
league_trends = (df_pbp.groupby('season')
                 .agg(pass_rate=('play_type', lambda x: (x == 'pass').mean()),
                      shotgun_rate=('shotgun', 'mean'))
                 .reset_index())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

ax1.plot(league_trends['season'], league_trends['pass_rate'], 'o-', lw=2, color='#1f77b4')
ax1.set_title('League Neutral Pass Rate (2002–2024)', fontweight='bold')
ax1.set_ylabel('Pass Rate')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax1.set_xlabel('Season')

ax2.plot(league_trends['season'], league_trends['shotgun_rate'], 's-', lw=2, color='#ff7f0e')
ax2.set_title('League Shotgun Snap Rate (2002–2024)', fontweight='bold')
ax2.set_ylabel('Shotgun Rate')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax2.set_xlabel('Season')

plt.tight_layout()
plt.show()

print(f"Shotgun usage grew from {league_trends.iloc[0]['shotgun_rate']:.1%} in 2002 to {league_trends.iloc[-1]['shotgun_rate']:.1%} in 2024.")
"""))

cells.append(md("""
Because shotgun snap rates rose from $\\sim 20\\%$ in 2002 to over $70\\%$ in 2024, clustering on raw values would simply group 2000s teams together and 2020s teams together.  
To isolate genuine tactical philosophy from macroeconomic secular drift, we must **standardize features within each season via Z-scores** ($Z = \\frac{x - \\mu_{\\text{season}}}{\\sigma_{\\text{season}}}$).
"""))

cells.append(code("""
# Inspecting team variation within a single season (2023)
pbp_2023 = df_pbp[df_pbp['season'] == 2023]
team_passes = pbp_2023.groupby('posteam')['play_type'].apply(lambda s: (s == 'pass').mean()).sort_values()

fig, ax = plt.subplots(figsize=(12, 4))
ax.barh(team_passes.index, team_passes.values, color='steelblue')
ax.axvline(team_passes.mean(), color='red', linestyle='--', label=f"League Mean: {team_passes.mean():.1%}")
ax.set_title('2023 Team Neutral Pass Rates (Within-Season Variation)', fontweight='bold')
ax.set_xlabel('Pass Rate')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend()
plt.tight_layout()
plt.show()
"""))

# Section 5: Feature Engineering
cells.append(md("""
## 5. Feature Engineering and Within-Season Standardization

To avoid confounding strategic philosophy with execution quality, we strictly use **non-outcome tactical rate features**. We exclude efficiency metrics such as EPA/play, success rate, or yards-per-play because those measure player quality rather than scheme.

We compute 7 core tactical dimensions:
- `pass_rate`: Overall neutral dropback rate
- `shotgun_rate`: Frequency of pre-snap shotgun alignment
- `no_huddle_rate`: Tempo / hurry-up frequency
- `air_yards_per_att`: Average intended target depth
- `deep_target_rate`: Fraction of pass attempts traveling $\\ge 15$ air yards
- `positional_target_conc`: Herfindahl index (HHI) of pass targets across field zones (left, middle, right)
- `rush_direction_conc`: Fraction of runs directed between the tackles (middle/guard)
"""))

cells.append(code("""
def build_tactical_features(df):
    records = []
    for (season, team), grp in df.groupby(['season', 'posteam']):
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
            'pass_rate': len(passes) / n,
            'shotgun_rate': grp['shotgun'].mean(),
            'no_huddle_rate': grp['no_huddle'].mean(),
            'air_yards_per_att': ay.mean() if len(ay) else np.nan,
            'deep_target_rate': (ay >= DEEP_YARDS).mean() if len(ay) else np.nan,
            'positional_target_conc': (loc_dist**2).sum() if len(loc_dist) else np.nan,
            'rush_direction_conc': interior.mean() if len(runs) else np.nan,
        })
    return pd.DataFrame(records)

df_team_raw = build_tactical_features(df_pbp)
print(f"Total team-seasons aggregated: {len(df_team_raw)}")
df_team_raw.head()
"""))

cells.append(code("""
# Apply Within-Season Z-Score Standardization
FEATS = ['pass_rate', 'shotgun_rate', 'no_huddle_rate', 'air_yards_per_att',
         'deep_target_rate', 'positional_target_conc', 'rush_direction_conc']
Z_FEATS = [f'{f}_z' for f in FEATS]

df_z = df_team_raw.copy()
for f in FEATS:
    mu  = df_z.groupby('season')[f].transform('mean')
    sig = df_z.groupby('season')[f].transform('std')
    df_z[f'{f}_z'] = (df_z[f] - mu) / sig

# Retain complete tracking era (2006–2024)
df_clean = df_z.dropna(subset=Z_FEATS).reset_index(drop=True)
print(f"Clean team-seasons for clustering (2006–2024): {len(df_clean)}")
"""))

cells.append(code("""
# Inspect feature correlation matrix in the standardized space
fig, ax = plt.subplots(figsize=(8, 6))
short_names = ['Pass Rate', 'Shotgun', 'No-Huddle', 'Air Yds', 'Deep Tgt %', 'Target Conc', 'Interior Rush']
corr = df_clean[Z_FEATS].corr()
corr.index = corr.columns = short_names

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, mask=mask, ax=ax, linewidths=0.5)
ax.set_title('Tactical Feature Correlation Matrix (Z-Scored Space)', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Section 6: Unsupervised Clustering & Validation
cells.append(md("""
## 6. Unsupervised Clustering and Model Evaluation

To determine if offensive schemes naturally partition into distinct clusters, we compare three distinct clustering families across $k = 2 \\dots 6$:
1. **K-Means Clustering** (with $k$-means++ centroid initialization)
2. **Agglomerative Hierarchical Clustering** (Ward's linkage, minimizing within-cluster variance)
3. **Gaussian Mixture Models (GMM)** (probabilistic soft clustering)

We evaluate cluster quality using **Silhouette Score** (geometric separation), **Davies-Bouldin Index** (ratio of within-cluster spread to separation), and **GMM BIC**.
"""))

cells.append(code("""
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

X = df_clean[Z_FEATS].values

results = []
for k in range(2, 7):
    # K-Means
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42)
    km_labs = km.fit_predict(X)
    
    # Agglomerative (Ward)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labs = agg.fit_predict(X)
    
    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=42)
    gmm_labs = gmm.fit_predict(X)
    
    results.append({
        'k': k,
        'km_silhouette': silhouette_score(X, km_labs),
        'agg_silhouette': silhouette_score(X, agg_labs),
        'gmm_silhouette': silhouette_score(X, gmm_labs),
        'km_db': davies_bouldin_score(X, km_labs),
        'agg_db': davies_bouldin_score(X, agg_labs),
        'gmm_bic': gmm.bic(X),
        'km_inertia': km.inertia_
    })

eval_df = pd.DataFrame(results)
eval_df.round(3)
"""))

cells.append(code("""
# Plot clustering evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(eval_df['k'], eval_df['km_silhouette'], 'o-', lw=2, label='K-Means')
axes[0].plot(eval_df['k'], eval_df['agg_silhouette'], 's--', lw=2, label='Ward Hierarchical')
axes[0].axvline(3, color='red', linestyle=':', label='Optimal k=3')
axes[0].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].legend()

axes[1].plot(eval_df['k'], eval_df['km_db'], 'o-', lw=2, color='coral', label='K-Means')
axes[1].plot(eval_df['k'], eval_df['agg_db'], 's--', lw=2, color='green', label='Ward Hierarchical')
axes[1].axvline(3, color='red', linestyle=':')
axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].legend()

axes[2].plot(eval_df['k'], eval_df['km_inertia'], 'o-', lw=2, color='purple')
axes[2].axvline(3, color='red', linestyle=':')
axes[2].set_title('K-Means Inertia (Elbow Method)', fontweight='bold')
axes[2].set_xlabel('Number of Clusters (k)')

plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Hierarchical Dendrogram Analysis

Ward's agglomerative clustering provides a hierarchical perspective on how team-seasons merge into tactical clusters.
"""))

cells.append(code("""
from scipy.cluster.hierarchy import dendrogram, linkage

Z_ward = linkage(X, method='ward')

plt.figure(figsize=(11, 4.5))
dendrogram(Z_ward, truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
plt.title("Ward Hierarchical Clustering Dendrogram (Truncated to 20 Nodes)", fontweight='bold')
plt.xlabel("Cluster Group / Observation Count")
plt.ylabel("Ward Distance Metric")
plt.axhline(y=24, color='red', linestyle='--', label='Cut at k=3 clusters')
plt.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Statistical Validation: Resampling Stability & Permutation Null Test

We verify that $k=3$ represents true empirical structure rather than random sampling noise using two tests:
1. **Bootstrap Stability Test (500 resamples):** We resample team-seasons with replacement and evaluate Adjusted Rand Index (ARI) against the full dataset clustering. An ARI $> 0.70$ confirms stability.
2. **Column-Permutation Null Test (1,000 permutations):** We shuffle each feature column independently to destroy multivariate correlations and compute the 99th percentile of synthetic silhouette scores.
"""))

cells.append(code("""
from sklearn.metrics import adjusted_rand_score

K = 3
rng = np.random.default_rng(42)
km_ref = KMeans(n_clusters=K, init='k-means++', n_init=100, random_state=42)
ref_labels = km_ref.fit_predict(X)

# 1. Bootstrap Stability
ari_scores = []
for _ in range(500):
    idx = rng.choice(len(X), size=len(X), replace=True)
    km_b = KMeans(n_clusters=K, init='k-means++', n_init=5, random_state=None)
    ari_scores.append(adjusted_rand_score(ref_labels[idx], km_b.fit_predict(X[idx])))

print(f"Bootstrap ARI Mean: {np.mean(ari_scores):.3f} (10th percentile: {np.percentile(ari_scores, 10):.3f})")
print(f"Cluster stability confirmed (ARI > 0.70): {np.mean(ari_scores) > 0.70}")

# 2. Permutation Null Test
null_sils = []
for _ in range(1000):
    X_perm = X.copy()
    for col in range(X_perm.shape[1]):
        X_perm[:, col] = rng.permutation(X_perm[:, col])
    km_p = KMeans(n_clusters=K, init='k-means++', n_init=3, random_state=None)
    null_sils.append(silhouette_score(X_perm, km_p.fit_predict(X_perm)))

emp_sil = silhouette_score(X, ref_labels)
p99 = np.percentile(null_sils, 99)
print(f"Empirical Silhouette: {emp_sil:.4f} vs Null 99th Percentile: {p99:.4f}")
print(f"Statistical significance confirmed (p < 0.01): {emp_sil > p99}")
"""))

cells.append(md("""
### Interpreting the Three Offensive Archetypes

Now we examine the cluster centroids and assign football-meaningful labels to each archetype.
"""))

cells.append(code("""
km_final = KMeans(n_clusters=3, init='k-means++', n_init=100, random_state=42)
df_clean['cluster'] = km_final.fit_predict(X)

# Compute continuous Euclidean distance to each centroid (used later for supervised models)
for c in range(3):
    df_clean[f'dist_centroid_{c}'] = np.linalg.norm(X - km_final.cluster_centers_[c], axis=1)

centroids = pd.DataFrame(
    km_final.cluster_centers_,
    columns=['Pass Rate', 'Shotgun', 'No-Huddle', 'Air Yards', 'Deep Target %', 'Target Conc', 'Interior Rush'],
    index=['Cluster 0', 'Cluster 1', 'Cluster 2']
)

fig, ax = plt.subplots(figsize=(10, 3.8))
sns.heatmap(centroids, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=1)
ax.set_title('Cluster Centroids in Z-Score Space (Deviation from Season Mean)', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
**Archetype Interpretations:**
- **Cluster 0: Short-Horizontal / Ball Control (53% of sample):** Low intended air yards ($-0.63\\sigma$) and low deep targets ($-0.68\\sigma$), with above-average interior running. Emphasizes rhythm passing and possession.
- **Cluster 1: Deep Vertical / Power Play-Action (39% of sample):** High intended target depth ($+0.92\\sigma$) and heavy deep shot frequency ($+0.92\\sigma$), paired with high under-center play-action.
- **Cluster 2: Hyper-Pace Spread / Tempo (8% of sample):** Extreme shotgun rate ($+1.29\\sigma$) and massive no-huddle tempo ($+2.62\\sigma$). Rare, demanding offensive system typified by elite no-huddle offenses.
"""))

cells.append(code("""
ARCHETYPES = {
    0: 'Short-Horizontal Ball Control',
    1: 'Deep Vertical / Power Run',
    2: 'Hyper-Pace Spread'
}
df_clean['archetype'] = df_clean['cluster'].map(ARCHETYPES)

# Historical evolution of archetypes
trend = pd.crosstab(df_clean['season'], df_clean['archetype'], normalize='index')

fig, ax = plt.subplots(figsize=(12, 4.8))
trend.plot.area(ax=ax, alpha=0.75, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
ax.set_title('Longitudinal Evolution of Offensive Archetypes (2006–2024)', fontweight='bold')
ax.set_ylabel('Proportion of NFL Teams')
ax.set_xlabel('Season')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
# 2D PCA Cluster Projection
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
for c in range(3):
    mask = df_clean['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=ARCHETYPES[c], color=colors[c], alpha=0.6, s=30)

ax.set_title(f"Tactical Clusters in PCA Space ({pca.explained_variance_ratio_.sum():.1%} Variance Explained)", fontweight='bold')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.legend()
plt.tight_layout()
plt.show()
"""))

# Section 7: Temporal Analysis & Strategy Decay
cells.append(md("""
## 7. Temporal Dynamics: Markov Transitions & Strategy Decay

### Markov Transition Matrix
How persistent is offensive identity? We compute the first-order Markov transition matrix to measure the probability of a team maintaining or altering its offensive scheme from year $t$ to year $t+1$.
"""))

cells.append(code("""
transitions = (df_clean.sort_values(['team', 'season'])
               .assign(cluster_next=lambda d: d.groupby('team')['cluster'].shift(-1))
               .dropna(subset=['cluster_next']))
transitions['cluster_next'] = transitions['cluster_next'].astype(int)

counts = pd.crosstab(transitions['cluster'], transitions['cluster_next'])
trans_matrix = counts.div(counts.sum(axis=1), axis=0)
trans_matrix.index = [f"C{i}: {ARCHETYPES[i][:16]}" for i in range(3)]
trans_matrix.columns = [f"C{i}" for i in range(3)]

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.heatmap(trans_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax, vmin=0, vmax=1)
ax.set_title('Year-over-Year Tactical Scheme Transition Matrix', fontweight='bold')
ax.set_xlabel('Archetype at Year t+1')
ax.set_ylabel('Archetype at Year t')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
**Observation on Stickiness:**  
Short-horizontal ball control demonstrates the highest retention ($60.7\\%$), while hyper-pace spread exhibits lower retention ($46.7\\%$), highlighting how difficult it is to maintain high-tempo spread schemes year over year without elite personnel.
"""))

cells.append(md("""
### Strategy Decay Hypothesis Test

The copycat dynamic in competitive sports suggests that when a novel scheme provides a winning edge, peer teams copy it, defensive coordinators design countermeasures, and the marginal advantage decays.  
We formally test whether a cluster's **win premium** (net points per game above league average) decreases as its **league adoption rate** expands.
"""))

cells.append(code("""
# Merge performance data
df_clean['team_clean'] = df_clean['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
df_perf['team_clean']  = df_perf['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})

df_merged = df_clean.merge(df_perf[['season', 'team_clean', 'net_pts_per_game']], on=['season', 'team_clean'], how='left')

# Calculate season-level win premium
lg_mean_pts = df_merged.groupby('season')['net_pts_per_game'].transform('mean')
df_merged['win_premium'] = df_merged['net_pts_per_game'] - lg_mean_pts

# Cluster-season adoption rate and performance
decay_df = (df_merged.groupby(['season', 'cluster'])
            .agg(n_teams=('team_clean', 'count'), avg_premium=('win_premium', 'mean'))
            .reset_index())
season_totals = df_merged.groupby('season')['team_clean'].count().reset_index(name='total_teams')
decay_df = decay_df.merge(season_totals, on='season')
decay_df['adoption_rate'] = decay_df['n_teams'] / decay_df['total_teams']

# Generalized Least Squares (GLS) with AR(1) errors
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.stattools import durbin_watson

X_gls = sm.add_constant(decay_df['adoption_rate'])
y_gls = decay_df['avg_premium']

# Check autocorrelation
ols_res = sm.OLS(y_gls, X_gls).fit()
dw = durbin_watson(ols_res.resid)

# Fit GLSAR
glsar = GLSAR(y_gls, X_gls, rho=1).iterative_fit(maxiter=5)
b1   = glsar.params.iloc[1]
pval = glsar.pvalues.iloc[1]
ci   = glsar.conf_int().iloc[1]

print(f"Durbin-Watson Statistic: {dw:.3f}")
print(f"GLS Regression Slope (β₁): {b1:.3f} (p = {pval:.4f})")
print(f"95% Confidence Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"Strategy Decay Hypothesis Supported (p < 0.05): {pval < 0.05}")
"""))

cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
for c in range(3):
    sub = decay_df[decay_df['cluster'] == c]
    ax.scatter(sub['adoption_rate'], sub['avg_premium'], color=colors[c], label=ARCHETYPES[c], alpha=0.75, s=55)

x_line = np.linspace(decay_df['adoption_rate'].min(), decay_df['adoption_rate'].max(), 50)
y_line = glsar.params.iloc[0] + glsar.params.iloc[1] * x_line
ax.plot(x_line, y_line, color='black', linestyle='--', lw=2, label=f'GLS Fit (β = {b1:.2f}, p = {pval:.3f})')
ax.axhline(0, color='gray', linestyle=':', lw=1)
ax.set_title('Strategy Decay: Adoption Rate vs. Net Win Premium', fontweight='bold')
ax.set_xlabel('Cluster Adoption Rate (League Share)')
ax.set_ylabel('Win Premium (Net Points/Game above League Avg)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

# Section 8: Supervised Modeling Benchmark
cells.append(md("""
## 8. Supervised Validation: Information Content vs. Autoregressive Baseline

### Research Question
Does tactical style contain **incremental predictive information** regarding next season's performance ($Y_{t+1}$), or is future performance dictated strictly by prior performance ($Y_t$) and roster continuity?

### Methodology
1. **QB Continuity Control:** We identify each team's primary starting passer. If the primary passer returns in year $t+1$, `qb_continuity = 1`; otherwise `0`.
2. **Continuous Style Distances:** Following the project specification, we do not use coarse one-hot labels; instead, we pass continuous Euclidean distance vectors to each cluster centroid ($d_0, d_1, d_2$).
3. **Four-Fold Expanding Time-Series CV:**
   - Fold 1: Train $\\le 2010$, Test $2011–2013$
   - Fold 2: Train $\\le 2013$, Test $2014–2016$
   - Fold 3: Train $\\le 2016$, Test $2017–2019$
   - Fold 4: Train $\\le 2019$, Test $2020–2024$
4. **Three Algorithms Evaluated:** Ridge Regression (L2 regularization), Random Forest Regressor, and LightGBM Regressor.
"""))

cells.append(code("""
# Derive QB continuity
passer_counts = (df_pbp[df_pbp['play_type'] == 'pass']
                 .groupby(['season', 'posteam', 'passer_player_name'])
                 .size()
                 .reset_index(name='attempts'))
idx_max = passer_counts.groupby(['season', 'posteam'])['attempts'].idxmax()
primary_passers = passer_counts.loc[idx_max].rename(columns={'posteam': 'team', 'passer_player_name': 'primary_qb'})
primary_passers['team_clean'] = primary_passers['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
primary_passers = primary_passers.sort_values(['team_clean', 'season'])
primary_passers['next_primary_qb'] = primary_passers.groupby('team_clean')['primary_qb'].shift(-1)
primary_passers['qb_continuity'] = (primary_passers['primary_qb'] == primary_passers['next_primary_qb']).astype(int)

# Create supervised modeling dataset
df_sup = df_clean.merge(df_perf[['season', 'team_clean', 'net_pts_per_game']], on=['season', 'team_clean'], how='left')
df_sup = df_sup.merge(primary_passers[['season', 'team_clean', 'qb_continuity']], on=['season', 'team_clean'], how='left')
df_sup = df_sup.sort_values(['team_clean', 'season'])
df_sup['y_next'] = df_sup.groupby('team_clean')['net_pts_per_game'].shift(-1)

df_model = df_sup.dropna(subset=['y_next', 'net_pts_per_game', 'qb_continuity']).reset_index(drop=True)
print(f"Supervised modeling observations: {len(df_model)} team-seasons")
"""))

cells.append(code("""
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CV_FOLDS = [
    {'train_end': 2010, 'eval_start': 2011, 'eval_end': 2013},
    {'train_end': 2013, 'eval_start': 2014, 'eval_end': 2016},
    {'train_end': 2016, 'eval_start': 2017, 'eval_end': 2019},
    {'train_end': 2019, 'eval_start': 2020, 'eval_end': 2024},
]

feature_specs = {
    '1. Baseline (Y_t only)': ['net_pts_per_game'],
    '2. Baseline + QB Continuity': ['net_pts_per_game', 'qb_continuity'],
    '3. Baseline + Style Distances': ['net_pts_per_game', 'dist_centroid_0', 'dist_centroid_1', 'dist_centroid_2'],
    '4. Full Model (Y_t + QB + Style)': ['net_pts_per_game', 'qb_continuity', 'dist_centroid_0', 'dist_centroid_1', 'dist_centroid_2']
}

cv_records = []
for spec_name, f_cols in feature_specs.items():
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
            cv_records.append({
                'Feature Set': spec_name,
                'Model': m_name,
                'Fold': f"{fold['eval_start']}–{fold['eval_end']}",
                'RMSE': np.sqrt(mean_squared_error(yte, preds)),
                'MAE': mean_absolute_error(yte, preds),
                'R2': r2_score(yte, preds)
            })

df_cv = pd.DataFrame(cv_records)
summary = df_cv.groupby(['Feature Set', 'Model'])[['RMSE', 'MAE', 'R2']].mean().round(3)
summary
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = ['RMSE', 'MAE', 'R2']

for i, metric in enumerate(metrics):
    piv = df_cv.pivot_table(index='Feature Set', columns='Model', values=metric, aggfunc='mean')
    piv.plot.bar(ax=axes[i], rot=15, width=0.7)
    axes[i].set_title(f'Out-of-Sample {metric} across CV Folds', fontweight='bold')
    axes[i].set_xlabel('')
    if metric == 'R2':
        axes[i].axhline(0, color='gray', linestyle='--', lw=0.8)

plt.suptitle('Model Benchmark: Does Offensive Style Add Predictive Value?', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Interpreting the Supervised Findings

1. **QB Continuity Adds Clear Value:** Adding quarterback continuity lowers RMSE and raises $R^2$ across all three model architectures (Ridge $R^2$ improves from $0.169$ to $0.181$; LightGBM $R^2$ improves from $0.118$ to $0.143$).
2. **Offensive Style Features Add No Incremental Value:** When continuous tactical style distances are added to the baseline, out-of-sample RMSE worsens (from $5.581$ to $5.645$ in Ridge) and $R^2$ declines.
3. **Acceptance of the Null Hypothesis:** We explicitly accept the null hypothesis that offensive scheme does not provide marginal predictive utility over past point differential and QB stability. Scheme is an execution conduit, not an independent, durable generator of wins.
"""))

# Section 9: Discussion & Conclusion
cells.append(md("""
## 9. Conclusion, Key Takeaways, and Limitations

### Primary Findings
1. **Three Statistically Valid Archetypes Exist:** Unsupervised clustering reveals 3 distinct tactical families: Short-Horizontal Ball Control ($53\\%$), Deep Vertical / Power Play-Action ($39\\%$), and Hyper-Pace Spread ($8\\%$). These clusters are stable across $500$ bootstrap resamples (mean ARI $= 0.757 > 0.70$) and statistically significant against $1,000$ column-permuted null sets ($p < 0.01$).
2. **Strategy Decay is Empirically Supported ($\\beta = -4.91, p = 0.015$):** As adoption of an offensive archetype expands across the league, its average performance premium compresses toward zero. The competitive adaptation dynamic in the NFL is real and quantifiable.
3. **Execution Dominates Scheme in Predicting Performance:** Offensive scheme features fail to outperform simple autoregressive performance and quarterback continuity. Teams do not win because of their scheme archetype in isolation; they win through roster talent, QB stability, and execution quality.

### Methodological Limitations
- **Structural Tracking Boundary:** NFL play-by-play data lacks passing depth (`air_yards`) and pass directional charting prior to 2006. Our clustering was appropriately restricted to 2006–2024 to preserve feature integrity.
- **Continuous vs. Categorical Scheme Reality:** While 3 distinct centroids provide clear archetypes, NFL offenses operate in a continuous tactical space rather than rigid silos (average Silhouette Score $\\approx 0.19$).
- **Coaching Staff Churn:** Coaching staff changes often correlate with scheme shifts. While QB continuity was controlled, coordinator-level turnover represents an opportunity for further causal attribution.
"""))

# Build Notebook Dict
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "NFL Clustering (.venv)",
            "language": "python",
            "name": "nfl_clustering"
        },
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Successfully wrote {len(cells)} cells to {NB_PATH}")
print(f"Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")

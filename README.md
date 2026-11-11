# NFL Offensive Scheme Clustering

A machine learning study analyzing 23 seasons (2002–2024) of NFL play-by-play data to identify offensive archetypes, trace their evolution, and test whether offensive scheme adds predictive value for future team performance.

---

## Research Questions

1. **What offensive archetypes exist across NFL team-seasons?** — Using unsupervised clustering to group teams by tactical style (e.g., Spread-Pace, Power-Under-Center, Deep-Vertical, Ball-Control).
2. **How have these styles evolved over time?** — Tracking year-to-year transitions via Markov matrices.
3. **Does offensive scheme predict future performance?** — Testing whether adding tactical cluster features improves regression accuracy over an autoregressive (past performance only) baseline.
4. **Strategy Decay Hypothesis** — Does a cluster's performance premium erode as more teams adopt that style?

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [nflverse / nflfastR](https://nflreadr.nflverse.com/) via `nflreadpy` |
| **Scope** | 2002–2024 NFL regular seasons |
| **Raw size** | ~1,050,000 play-by-play records, 372 columns |
| **Aggregated** | 736 team-season observations (32 teams × 23 seasons) |

### Key Engineered Features *(all filtered to neutral game scripts: WP 20–80%, not in final 2 min of each half)*

| Feature | Description |
|---|---|
| Neutral Early-Down Pass Rate | % of 1st/2nd down plays that are passes |
| Shotgun Formation Frequency | % of plays run from shotgun |
| No-Huddle Tempo Rate | % of plays run without a huddle |
| Intended Air Yards per Attempt | Average depth of target per pass attempt |
| Deep Target Allocation Rate | % of targets thrown ≥15 yards downfield |
| Positional Target Concentration | Target share split across RBs, TEs, WRs |
| Rushing Directional Concentration | Interior vs. perimeter run distribution |

> All features are **within-season Z-score standardized** to remove league-wide secular trends (e.g., the league-wide shotgun rate rose from ~20% in 2002 to ~70% in 2024 — raw rates would cluster eras, not styles).

### Target Variable

**Y_{t+1}** = (Points Scored_{t+1} − Points Allowed_{t+1}) / Games Played_{t+1}  
*(next-season net point differential per game)*

---

## ML Pipeline

### Phase 1 — Data Ingestion & Cleaning
- Load play-by-play data via `nflreadpy`
- Remove kneels, spikes, special teams plays, dead-ball penalties
- Reclassify QB scrambles as called dropbacks (preserving true play-calling intent)
- Filter to neutral game scripts (WP 20–80%, not in final 2 min of each half)

### Phase 2 — Feature Engineering & EDA
- Aggregate plays to team-season level (736 observations)
- Construct 7 tactical rate features
- Within-season Z-score standardization
- Exploratory visualizations of feature distributions and temporal trends

### Phase 3 — Unsupervised Clustering
- **Algorithms:** K-Means (k-means++ init, 100 restarts), Gaussian Mixture Models (full/tied covariance), Ward Hierarchical Clustering
- **Cluster validation:** Silhouette Score, Davies-Bouldin Index, BIC
- **Stability:** Bootstrap test (500 resamples, target ARI > 0.70); Null permutation test (1,000 permuted datasets, empirical Silhouette must exceed 99th percentile)
- **Temporal analysis:** Markov transition matrices for year-to-year style persistence
- **Strategy Decay:** Generalized Least Squares regression of cluster win premium on adoption rate across seasons

### Phase 4 — Supervised Regression
- **Baseline:** Autoregressive model (past Y_t only)
- **Style-augmented:** Y_t + tactical cluster centroid distance vectors
- **Models:** Ridge Regression, Random Forest, LightGBM
- **Validation:** Expanding-window time-series cross-validation (prevents temporal data leakage)

| Fold | Train | Evaluate |
|---|---|---|
| 1 | 2002–2010 | 2011–2013 |
| 2 | 2002–2013 | 2014–2016 |
| 3 | 2002–2016 | 2017–2019 |
| 4 | 2002–2019 | 2020–2024 |

- **Metrics:** Out-of-sample RMSE, MAE, R²

---

## Project Structure

```
.
├── nfl_offensive_clustering.ipynb   # Main deliverable — full analysis notebook
├── README.md
└── .gitignore
```

> Raw data is fetched programmatically via `nflreadpy` and is not committed to this repository.

---

## Key Libraries

| Library | Purpose |
|---|---|
| `nflreadpy` | NFL play-by-play data ingestion |
| `polars` / `pandas` | Data manipulation |
| `numpy` / `scipy` | Numerical operations, Z-scoring |
| `scikit-learn` | Clustering, Ridge, Random Forest, CV |
| `lightgbm` | Gradient boosted regression |
| `statsmodels` | GLS regression for decay hypothesis |
| `matplotlib` / `seaborn` | Visualization |

---

## Known Pitfalls & Mitigations

| Pitfall | Mitigation |
|---|---|
| QB injury / departure confounds Y_{t+1} | Binary QB continuity control variable (starter played ≥12 games in both seasons t and t+1) |
| League-wide era trends conflate styles | Within-season Z-score standardization |
| Temporal data leakage | Expanding-window CV — never train on future seasons |
| Centroid distance collinearity in regression | Ridge regularization |
| Survivorship bias in Markov matrices | Explicitly documented in notebook narrative |

---

## Course

**PRINCPLS OF MACHINE LEARN I — Fall 2026 (55415)**  
Final project due: **November 19, 2026**

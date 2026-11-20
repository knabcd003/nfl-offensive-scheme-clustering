# NFL Offensive Scheme Clustering

A machine learning study analyzing 23 seasons (2002–2024) of NFL play-by-play data to identify offensive archetypes, trace their evolution, and test whether offensive scheme adds predictive value for future team performance.

---

## Research Questions

1. **What offensive archetypes exist across NFL team-seasons?** — Using unsupervised clustering to group teams by tactical style. The final K-Means model finds three: Short-Depth Passing, Shotgun / No-Huddle, and Deep Vertical Passing.
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
| **Aggregated** | 736 team-seasons (32 × 23); 608 from 2006–2024 used for clustering (air yards and pass location are untracked before 2006); 576 with a next-season outcome used for regression |

### Key Engineered Features *(all filtered to neutral game scripts: WP 20–80%, not in final 2 min of each half)*

| Feature | Description |
|---|---|
| Neutral Dropback Rate | % of neutral plays that are QB dropbacks (pass attempts, sacks, scrambles) |
| Shotgun Formation Frequency | % of plays run from shotgun |
| No-Huddle Tempo Rate | % of plays run without a huddle |
| Intended Air Yards per Attempt | Average depth of target per pass attempt |
| Deep Target Allocation Rate | % of targets thrown ≥15 yards downfield |
| Target Location Concentration | Herfindahl index (HHI) of pass targets across left / middle / right |
| Middle Run Share | % of runs charted up the middle (nflverse `run_location == 'middle'`) |

> All features are **within-season Z-score standardized** to remove league-wide secular trends (e.g., the league-wide shotgun rate rose from ~10% in 2002 to ~67% in 2024 — raw rates would cluster eras, not styles).

### Target Variable

**Y_{t+1}** = (Points Scored_{t+1} − Points Allowed_{t+1}) / Games Played_{t+1}  
*(next-season net point differential per game)*

---

## ML Pipeline

### Phase 1 — Data Ingestion & Cleaning
- Load play-by-play data via `nflreadpy`
- Remove kneels, spikes, special teams plays, dead-ball penalties
- Measure pass rate with `qb_dropback`, which already counts QB scrambles as called dropbacks (`play_type == 'pass'` would treat scrambles as runs)
- Filter to neutral game scripts (WP 20–80%, not in final 2 min of each half)

### Phase 2 — Feature Engineering & EDA
- Aggregate plays to team-season level (736 observations)
- Construct 7 tactical rate features
- Restrict to 2006–2024 (608 team-seasons), the seasons with air-yards and pass-location tracking
- Within-season Z-score standardization
- Exploratory visualizations of feature distributions and temporal trends

### Phase 3 — Unsupervised Clustering
- **Algorithms:** K-Means (k-means++ init; 50 restarts when comparing k, 100 for the final model), Gaussian Mixture Models (full covariance), Ward Hierarchical Clustering, each for k = 2–6
- **Cluster validation:** Silhouette Score, Davies-Bouldin Index, BIC
- **Stability:** Bootstrap test (500 resamples, target ARI > 0.70); Null permutation test (1,000 permuted datasets, empirical Silhouette must exceed 99th percentile)
- **Temporal analysis:** Markov transition matrices for year-to-year style persistence
- **Strategy Decay:** GLS regression with AR(1) errors (GLSAR) of each archetype's average win premium on its adoption rate, one row per archetype per season

### Phase 4 — Supervised Regression
- **Baseline:** Autoregressive model (past Y_t only)
- **Style-augmented:** Y_t + distances to the three K-Means centroids (K-Means refit inside each CV fold on training seasons only)
- **Retrospective comparison:** the same two feature sets plus realized QB continuity, which is only known after season t+1
- **Models:** Ridge Regression, Random Forest, LightGBM
- **Validation:** Expanding-window time-series cross-validation (prevents temporal data leakage)

| Fold | Train | Evaluate |
|---|---|---|
| 1 | 2006–2010 | 2011–2013 |
| 2 | 2006–2013 | 2014–2016 |
| 3 | 2006–2016 | 2017–2019 |
| 4 | 2006–2019 | 2020–2023 |

*Seasons are feature seasons; each row's target is the following season. Fold 4 stops at 2023 because 2024 has no next-season outcome yet.*

- **Metrics:** Out-of-sample RMSE, MAE, R²
- **Hyperparameter Tuning (Robustness Check):** Nested time-series CV within outer training folds (tuning Ridge $\alpha$, RF tree depth/leaf size, LightGBM learning rate/leaves/regularization) confirming that the null finding is invariant to model capacity and scaling
- **Error Analysis & Diagnostics:** Out-of-sample residual breakdown across QB continuity (+21.6% MAE increase when starting QB changes), offensive archetypes (verifying zero systematic bias across schemes), and qualitative audit of the 10 largest prediction outliers (7/10 driven by QB injuries or transitions)

---

## Project Structure

```
.
├── nfl_offensive_clustering.ipynb   # Main deliverable — full analysis notebook
├── README.md
├── PROCESS_NOTES.md                 # Methodology decisions and changelog
├── scripts/                         # Exploration and verification scripts
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
| QB injury / departure confounds Y_{t+1} | Binary QB continuity: same primary passer (most neutral pass plays, matched by player ID) in seasons t and t+1; used only in the retrospective comparison because it is known only after t+1 |
| League-wide era trends conflate styles | Within-season Z-score standardization |
| Temporal data leakage | Expanding-window CV — never train on future seasons |
| Centroid distance collinearity in regression | Ridge regularization |
| Broken season sequences in Markov transitions | Transitions require consecutive calendar seasons (`season_next == season + 1`) |

---

## Course

**PRINCPLS OF MACHINE LEARN I — Fall 2026 (55415)**  
Final project due: **November 19, 2026**

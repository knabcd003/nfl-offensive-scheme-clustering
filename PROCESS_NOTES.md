# Project Technical Evolution and Methodology Notes

**Project:** NFL Offensive Scheme Clustering and Strategic Decay  
**Course:** Principles of Machine Learning I — Fall 2026  
**Authors:** Karthik N., Aneek S., et al.  

---

## Overview

This document chronicles the technical, statistical, and data engineering decisions made throughout the lifecycle of this project. As our analysis progressed from initial concept to empirical implementation, working directly with raw micro-level event data revealed several critical domain realities, tracking boundaries, and modeling constraints. 

Each section details a specific technical divergence from our initial setup: what the data showed, the statistical or methodological problem identified, and how our data pipeline and models were adjusted to resolve it.

---

### 1. Dataset Scope and the 2006 Optical Tracking Boundary

* **Initial Plan:** Evaluate all 23 NFL seasons from 2002 to 2024 (736 potential team-seasons across 32 franchises).
* **What We Noticed:** When standardizing tactical features across seasons and handling missing data, exactly 128 team-seasons were dropped. Auditing missingness field-by-field across all 23 seasons revealed that intended target depth (`air_yards`) and pass directional charting (`pass_location`) were **$100\%$ unrecorded from 2002 to 2005**. Standardized optical tracking and field-location charting only began uniformly across all 32 NFL stadiums in **2006** ($4\text{ seasons} \times 32\text{ teams} = 128$).
* **Technical Resolution:** Rather than using artificial imputation or falling back to crude, non-depth proxies for the early 2000s, we anchored our tactical feature space strictly to **2006–2024** ($19$ completed seasons, $608$ team-seasons). This guaranteed that every observation possessed complete, uniform spatial tracking data.

---

### 2. Isolating Strategic Intent from Macroeconomic Secular Drift

* **Initial Plan:** Cluster offenses on absolute tactical frequencies (e.g., raw pass rate, raw shotgun rate, raw no-huddle rate).
* **What We Noticed:** Inspecting league-wide historical trends revealed massive secular drift: neutral-situation shotgun snap rates surged from under $10\%$ in 2002 to nearly $70\%$ by 2024, and pass rates climbed from $54\%$ to over $60\%$. Clustering on raw rates resulted in unsupervised algorithms grouping teams primarily by *era* rather than tactical philosophy.
* **Technical Resolution:** We implemented **within-season Z-score standardization**:
  $$Z_{i,t} = \frac{x_{i,t} - \mu_{\text{season}, t}}{\sigma_{\text{season}, t}}$$
  In this standardized space, a positive value indicates tactical divergence relative to contemporary peers that specific season. A $50\%$ shotgun rate in 2003 represents a heavy spread outlier ($+2.8\sigma$), whereas a $50\%$ shotgun rate in 2023 represents a run-heavy set ($-1.9\sigma$). This decoupled organizational tactical identity from league-wide macro drift.

---

### 3. Pure Non-Outcome Features vs. Efficiency Conflation

* **Initial Plan:** Include offensive efficiency metrics such as Expected Points Added (EPA) per play, success rate, or yards per play alongside formation frequencies to characterize offensive systems.
* **What We Noticed:** Efficiency metrics introduced a severe confounder: they conflated *how an offense chose to play* (strategic scheme) with *how well they executed* (player talent and execution). A team running an identical West Coast scheme would cluster in completely different locations depending on whether their quarterback was an MVP candidate or a backup.
* **Technical Resolution:** We strictly restricted our feature space to **non-outcome tactical deployment choices**:
  1. `pass_rate` (called dropback tendency)
  2. `shotgun_rate` (pre-snap formation alignment)
  3. `no_huddle_rate` (tempo and hurry-up frequency)
  4. `air_yards_per_att` (intended downfield aggression)
  5. `deep_target_rate` (proportion of throws traveling $\ge 15$ yards)
  6. `positional_target_conc` (spatial dispersion of targets across field zones via HHI)
  7. `rush_direction_conc` (proportion of runs directed between the tackles)

---

### 4. Accounting for QB Scrambles as Called Dropbacks

* **Initial Plan:** Calculate offensive pass rate using `play_type == 'pass'`.
* **What We Noticed:** In official NFL box scores and raw play-by-play logs, quarterback scrambles are recorded with `play_type == 'run'`. Filtering on `play_type == 'pass'` inadvertently treated all 9,448 QB scrambles in our neutral sample as designed running plays. Teams with mobile quarterbacks (e.g., Michael Vick, Cam Newton, Lamar Jackson, Josh Allen) were artificially skewed toward appearing like run-heavy ground schemes, even though the called play was an intended pass.
* **Technical Resolution:** We reclassified all plays where `qb_scramble == 1` as dropbacks (`qb_dropback = 1`) and defined our pass-rate feature as `grp['qb_dropback'].fillna(0).mean()`. This properly attributed intended passing plays to passing volume regardless of whether the play ended in a pass attempt, a sack, or a scramble.

---

### 5. Feature Distribution Auditing, Skewness, and Extreme Tactical Outliers

* **Initial Plan:** Standardize the 7 tactical features and proceed directly to clustering without inspecting their empirical distributions or searching for multivariate outliers.
* **What We Noticed:** Upon deeper exploratory analysis, we audited the parametric shapes, central tendencies, and extremes of all 7 features across the 608 team-seasons (2006–2024):
  1. *Severe tempo skewness:* `no_huddle_rate` exhibited extreme right-skewness ($\text{skew} = 3.50$). While the median NFL offense ran no-huddle on just $3.4\%$ of neutral plays, a handful of radical tempo innovators pushed rates above $60\%–70\%$.
  2. *Near-Gaussian passing metrics:* In contrast, `air_yards_per_att` ($\text{skew} = 0.11$, mean $8.05$ yards) and `pass_rate` ($\text{skew} = -0.21$, mean $57.7\%$) displayed smooth, symmetrical distributions.
  3. *Identifying historical tactical anomalies ($|Z| > 3.0$):* Inspecting standardized boxplots revealed that statistical outliers directly corresponded to identifiable, extreme schemes in NFL history:
     * *Pace outliers:* 2013–2015 Philadelphia Eagles under Chip Kelly ($+3.50\sigma$ to $+3.70\sigma$ tempo), 2010 Indianapolis Colts under Peyton Manning ($+4.58\sigma$ tempo), and 2019–2022 Arizona Cardinals under Kliff Kingsbury ($+3.66\sigma$ to $+4.09\sigma$ tempo).
     * *Formation outliers:* 2007 New England Patriots ($+3.26\sigma$ shotgun rate), pioneering full-time spread formations during their 16–0 season.
     * *Run-heavy outliers:* 2011 Denver Broncos with Tim Tebow ($39.8\%$ pass rate, $-3.18\sigma$), running a college-style option ground attack.
     * *Interior power run outliers:* 2008–2009 Jacksonville Jaguars with Maurice Jones-Drew and Fred Taylor ($+3.05\sigma$ interior rush share).
* **Technical Resolution:** We incorporated 7-panel distribution histograms with KDE curves, summary statistics tables (reporting mean, standard deviation, median, IQR, and skewness), and standardized boxplots into the data exploration section. This confirmed that our features are bounded and physiologically sound (no negative air yards or rates outside $[0, 1]$), while explaining the geometric mechanism of downstream clustering: the heavy right-tail of tempo naturally separates the rare "Shotgun / No-Huddle" archetype (Cluster 1, $\sim 8\%$) from the broader league.

---

### 6. Regular-Season Target Normalization, Franchise Aliases, and Exact Calendar Joins

* **Initial Plan:** Aggregate seasonal team performance (net points per game) from the full schedule table and use pandas positional `shift(-1)` to align year $t$ with year $t+1$.
* **What We Noticed:** 
  1. *Playoff distortion:* Unfiltered schedule tables included postseason games. Playoff games are played exclusively by top-tier teams against elite opponents, penalizing playoff teams with extra games against top competition and distorting their per-game net points relative to non-playoff teams.
  2. *Franchise relocations and aliases:* Historical franchise moves (St. Louis to Los Angeles, San Diego to Los Angeles, Oakland to Las Vegas) and shifting abbreviation conventions (`JAC`/`JAX`, `WSH`/`WAS`) broke cross-season merges.
  3. *Positional shift gaps:* Relying on `shift(-1)` within a groupby assumes perfectly continuous, unbroken season sequences. If a team row was missing for a single season, `shift(-1)` silently paired season $t$ with season $t+2$.
* **Technical Resolution:**
  * Filtered schedules strictly to completed regular-season games (`game_type == 'REG'`).
  * Unified all franchise identifiers across both play-by-play and schedule tables (`TEAM_ALIASES = {'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV', 'JAC': 'JAX', 'WSH': 'WAS'}`).
  * Replaced positional row shifting with explicit calendar year checks: in the Markov transition table, we added `.query('season_next == season + 1')`, and in supervised modeling, we explicitly merged on `target_season = season + 1` with `validate='one_to_one'`.

---

### 7. Multi-Algorithm Clustering and Statistical Significance Testing

* **Initial Plan:** Run K-Means, select $k$ via an elbow plot, and proceed to downstream modeling.
* **What We Noticed:** Clustering algorithms can partition random continuous data into visually distinct clusters. We needed to verify whether these tactical groupings represented true geometric structure or arbitrary divisions of a continuous space.
* **Technical Resolution:** 
  1. Evaluated three distinct clustering families across $k=2\dots 6$: K-Means, Ward Agglomerative Hierarchical, and Gaussian Mixture Models (GMM), tracking Silhouette Score, Davies-Bouldin Index, and GMM BIC.
  2. Plotted a truncated Ward linkage dendrogram to inspect the hierarchical merge distances.
  3. Executed a **500-sample bootstrap stability test**, confirming cluster stability with a mean Adjusted Rand Index (ARI) of $0.709$ (exceeding the standard $0.70$ stability threshold).
  4. Executed a **1,000-iteration column-permutation null test**: shuffling feature columns independently to destroy correlation structure confirmed that our empirical silhouette score ($0.1864$) significantly exceeded the 99th percentile of synthetic null noise ($0.1300$, $p < 0.01$).

---

### 8. Archetype Index Realignment and Label Verification

* **What We Noticed:** After fixing the scramble dropback calculation in the feature matrix, running `KMeans(n_clusters=3, random_state=42)` produced a slight index permutation in the resulting cluster labels: Cluster 0 remained the large short-passing cluster ($52.5\%$), but Cluster 1 was assigned to the high-tempo shotgun cluster ($8.1\%$), and Cluster 2 was assigned to the deep vertical passing cluster ($39.5\%$).
* **Technical Resolution:** We verified the coordinates of the new cluster centroids and updated the dictionary mapping to ensure descriptive accuracy:
  ```python
  ARCHETYPES = {
      0: 'Short-Depth Passing',
      1: 'Shotgun / No-Huddle',
      2: 'Deep Vertical Passing'
  }
  ```
  This ensured that all longitudinal area plots, transition heatmaps, and strategy decay regressions accurately reflected the mathematical centroid coordinates.

---

### 9. Preventing Unsupervised Data Leakage in Cross-Validation

* **Initial Plan:** Fit a single `KMeans(n_clusters=3)` on all 608 team-seasons (2006–2024), compute Euclidean distances to those three centroids, and pass those pre-computed distance columns into the expanding-window cross-validation loop.
* **What We Noticed:** This introduced **unsupervised lookahead leakage**: when evaluating models on 2011–2013 data, the cluster centroids were calculated from data that included the 2020s. Test folds were being projected onto geometric coordinates informed by future football.
* **Technical Resolution:** We moved the unsupervised fitting inside each expanding CV fold:
  ```python
  km_fold = KMeans(n_clusters=3, init='k-means++', n_init=100, random_state=42)
  km_fold.fit(tr[Z_FEATS])
  tr[style_cols] = km_fold.transform(tr[Z_FEATS])
  te[style_cols] = km_fold.transform(te[Z_FEATS])
  ```
  Centroids are learned exclusively on historical training seasons ($\le t$), and test seasons are transformed using only those historical coordinates.

---

### 10. QB Continuity Controls: Forecasting vs. Retrospective Benchmark

* **Initial Plan:** Treat quarterback stability as a standard forward-looking feature alongside offensive style when predicting next season's performance ($Y_{t+1}$), identifying QBs by their displayed text names.
* **What We Noticed:** 
  1. *Player ID collisions:* Merging passers on string names like `passer_player_name` caused collisions across players sharing initials and surnames.
  2. *Ex-ante vs. Ex-post:* At the end of season $t$, a team's offensive style from season $t$ is known before season $t+1$ starts. However, you cannot know with certainty who will be the *realized* primary starting passer in season $t+1$ (due to unexpected injuries or depth chart changes). Realized QB continuity is technically an **ex-post retrospective control**, not an ex-ante forecast feature.
  3. *Fold boundary:* In our initial CV setup, Fold 4 had `eval_end: 2024`. Because 2025 has not yet occurred, 2024 rows had no target $y_{t+1}$ and were dropped.
* **Technical Resolution:** 
  * Identified each team's primary passer in the neutral sample using unique `passer_player_id` keys.
  * Separated our supervised evaluation into two distinct benchmarks:
    1. **Pure Forecasting Comparison (Ex-ante inputs only):** Comparing past performance ($Y_t$) against past performance plus style ($Y_t$ + Style). Adding style worsened Ridge RMSE from $5.581$ to $5.666$ ($R^2$ dropped from $0.169$ to $0.145$). Tactical style does not improve forward-looking forecasts.
    2. **Retrospective Explanatory Comparison (Controlling for realized QB health):** Comparing $Y_t$ + realized QB continuity against that baseline plus style. Realized QB continuity noticeably improved accuracy (Ridge RMSE dropped to $5.552$, $R^2 = 0.179$), but adding style once again degraded accuracy (Ridge RMSE rose to $5.634$).
  * Bounded Fold 4 `eval_end` to 2023 (whose target season is 2024).

---

### 11. Econometric Modeling of Strategy Decay and Serial Autocorrelation

* **Initial Plan:** Regress each archetype's annual win premium on its league-wide adoption rate via Ordinary Least Squares (OLS) to test if copycat adoption erodes competitive advantage.
* **What We Noticed:** Ordinary Least Squares (OLS) on stacked time-series panels violates standard independence assumptions due to serial correlation in the error terms, which can artificially inflate $t$-statistics. Additionally, `statsmodels.regression.linear_model.GLSAR` threw a runtime `KeyError: 1` when extracting parameters because `sm.add_constant()` produces a pandas Series with named indices (`'const'`, `'adoption_rate'`).
* **Technical Resolution:** 
  * Fixed parameter extraction using positional lookups (`gls.params.iloc[1]`).
  * Checked for autocorrelation using the **Durbin-Watson statistic** ($\text{DW} = 2.30$).
  * Estimated the regression using **Generalized Least Squares with AR(1) errors (GLSAR)**.
  * The resulting slope was negative and statistically significant:
    $$\beta_1 = -5.467, \quad p = 0.013, \quad 95\%\text{ CI: } [-9.733, -1.202]$$
  This empirically verified our Strategy Decay hypothesis: as the league-wide adoption of an offensive archetype expands, its performance premium systematically compresses toward zero as defensive coordinators design schematic countermeasures.

---

### 12. Acceptance of the Null Hypothesis in Supervised Modeling

* **Initial Expectation:** Tactical scheme archetypes were hypothesized to provide incremental predictive information over past point differential when forecasting future team performance.
* **What the Data Showed:** Across Ridge Regression, Random Forest, and LightGBM, adding tactical scheme features consistently worsened out-of-sample error and lowered $R^2$ compared to past performance and quarterback stability alone.
* **Technical Resolution:** Rather than altering hyperparameters or adding arbitrary non-linear interaction terms to force an artificial in-sample improvement, we **explicitly accepted the null hypothesis**. In the NFL, offensive scheme is an operational conduit through which roster talent and quarterback execution are expressed, not an independent generator of sustainable competitive advantage. Documenting why scheme fails to predict future performance—alongside our finding that winning schemes decay as they are copied—represents an empirical finding consistent with domain reality.

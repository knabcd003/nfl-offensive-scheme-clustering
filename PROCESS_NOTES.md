# Project Evolution and Methodology Notes

**Project:** NFL Offensive Scheme Clustering and Strategic Decay  
**Course:** Principles of Machine Learning I — Fall 2026  
**Authors:** Karthik N., Aneek S., et al.  

---

## Overview

When we initially submitted our project proposal, we had a high-level theoretical concept: take 23 years of NFL play-by-play data, cluster teams into tactical offensive schemes, and predict future team success to see if offensive scheme serves as an independent, durable competitive advantage.

However, moving from an abstract proposal to actual empirical implementation required confronting the messiness of real-world sports data. In our initial exploratory runs, we quickly realized that a naive "one-shot" pipeline produced results that were performative and fragile—large monolithic code cells doing ten operations at once, uninspected data drops, and statistical leakage. 

We stepped back, stripped the workflow down to standalone exploratory Python scripts, and rebuilt the entire project step-by-step. Each time we observed something unexpected in the data, we adjusted our methodology to ensure mathematical integrity. 

This document chronicles the full evolution of our project: what we originally planned, what we noticed when digging into the data, the mistakes we caught, how we fixed them, and why our final pipeline looks the way it does.

---

### 1. From Monolithic "AI Notebook" to Ground-Up Modular Engineering

* **Initial State:** Our very first draft of the Jupyter notebook attempted to execute the entire analysis in massive, all-in-one code blocks with polished executive-summary markdown.
* **What We Noticed:** Upon critical review, the notebook felt like a "one-shot LARP"—it favored showing off over scientific substance. Large cells obscured data issues, hid potential bugs, and made it impossible to trace the logical progression of the machine learning pipeline.
* **What We Changed:** We nuked the monolithic notebook completely. Before touching Jupyter again, we adopted an engineering rule: **explore and verify everything in standalone Python scripts first** (`scripts/10_...` through `scripts/18_...`). Once each data transformation, statistical test, and model benchmark was verified in code, we reconstructed the notebook from scratch into **42 small, single-purpose cells** (22 code, 20 markdown). Each cell does exactly one task (e.g., check missingness, plot drift, fit one model family), narrated in a direct, student-authentic voice explaining what we were about to do and what the data showed.

---

### 2. Dataset Scope and the 2006 Optical Tracking Boundary

* **Initial Plan:** We originally planned to evaluate all 23 seasons from 2002 to 2024 (736 potential team-seasons across 32 franchises).
* **What We Noticed:** When we first ran within-season standardization and dropped rows with missing values, our sample dropped from 736 to 608 team-seasons—exactly 128 observations were missing. We didn't want to accept a black-box data drop, so we wrote `scripts/11_inspect_air_yards_and_locations.py` to audit missingness field-by-field across all 23 seasons. We discovered that intended target depth (`air_yards`) and pass directional charting (`pass_location`) were **$100\%$ unrecorded from 2002 to 2005**. Standardized optical tracking and field charting only began uniformly across all 32 NFL stadiums in **2006** ($4\text{ seasons} \times 32\text{ teams} = 128$).
* **What We Changed:** Rather than using artificial imputation or falling back to crude, non-depth proxies for the early 2000s, we anchored our tactical clustering strictly to **2006–2024** ($19$ completed seasons, $608$ team-seasons). This guaranteed that every team-season in our feature space had complete, uniform tracking data across all spatial dimensions.

---

### 3. Isolating Strategic Intent from Macroeconomic Secular Drift

* **Initial Plan:** We initially considered clustering on absolute frequencies (e.g., raw pass rate, raw shotgun rate, raw no-huddle rate).
* **What We Noticed:** When inspecting league-wide historical trends, we saw massive macroeconomic drift: neutral-situation shotgun snap rates surged from under $10\%$ in 2002 to nearly $70\%$ by 2024, and pass rates steadily climbed from $54\%$ to over $60\%$. If we clustered on raw rates, an unsupervised algorithm would simply cluster on *time periods*—grouping early-2000s offenses together and 2020s offenses together—rather than identifying genuine tactical philosophy.
* **What We Changed:** In order to decouple organizational style from secular league-wide evolution, we implemented **within-season Z-score standardization**:
  $$Z_{i,t} = \frac{x_{i,t} - \mu_{\text{season}, t}}{\sigma_{\text{season}, t}}$$
  In this standardized space, a positive value always indicates that a team utilized a tactical concept *more than its contemporary peers that specific season*. A $50\%$ shotgun rate in 2003 represented a revolutionary spread pioneer ($+2.8\sigma$), whereas a $50\%$ shotgun rate in 2023 represented a retro, run-heavy offense ($-1.9\sigma$). This transformation allowed us to track the evolution of schematic archetypes across eras on a common, relative scale.

---

### 4. Pure Non-Outcome Features vs. Efficiency Conflation

* **Initial Plan:** We initially considered including offensive efficiency metrics such as Expected Points Added (EPA) per play, success rate, or yards per play to characterize offensive systems.
* **What We Noticed:** During early feature engineering, we realized that including efficiency metrics introduced a fundamental confounder: it blended *how an offense chose to play* (strategic scheme) with *how well they executed* (player talent and execution). A team running an identical West Coast offense would appear in completely different clusters depending on whether their quarterback was Patrick Mahomes or a backup.
* **What We Changed:** In order to isolate pure tactical philosophy, we strictly restricted our feature set to **non-outcome tactical deployment choices**:
  1. `pass_rate` (called dropback tendency)
  2. `shotgun_rate` (pre-snap formation alignment)
  3. `no_huddle_rate` (tempo and hurry-up frequency)
  4. `air_yards_per_att` (intended downfield aggression)
  5. `deep_target_rate` (proportion of throws traveling $\ge 15$ yards)
  6. `positional_target_conc` (spatial dispersion of targets across field zones via HHI)
  7. `rush_direction_conc` (proportion of runs directed between the tackles)

---

### 5. Accounting for QB Scrambles as Called Dropbacks

* **Initial Plan:** In our first pass at aggregating team play-calling, we calculated pass rate using `play_type == 'pass'`.
* **What We Noticed:** In official NFL box score data and standard play-by-play logs, quarterback scrambles are recorded with `play_type == 'run'`. By filtering strictly on `play_type == 'pass'`, we were inadvertently counting all 9,448 QB scrambles in our neutral sample as designed running plays. Teams with highly mobile quarterbacks (like Michael Vick, Cam Newton, Lamar Jackson, or Josh Allen) were artificially skewed toward appearing like run-heavy ground schemes, even though the coordinator had called a pass play that broke down into a scramble.
* **What We Changed:** In order to measure actual play-calling intent, we reclassified all plays where `qb_scramble == 1` as dropbacks (`qb_dropback = 1`) and defined our pass-rate feature as `grp['qb_dropback'].fillna(0).mean()`. This properly attributed intended passing plays to passing volume regardless of whether the play ended in a throw, a sack, or a scramble.

---

### 6. Regular-Season Target Normalization, Franchise Aliases, and Exact Calendar Joins

* **Initial Plan:** When aggregating seasonal team performance (net points per game) from game schedules, our initial pipeline aggregated all completed games in the schedule table and used pandas row-positional `shift(-1)` to align year $t$ with year $t+1$.
* **What We Noticed:** We noticed three separate data integrity issues:
  1. *Playoff distortion:* Schedule tables included postseason games. Playoff games are only played by top-tier teams against other elite opponents. If a 13–3 team with a $+10.0$ net points per game regular season lost a blowout playoff game, their full-season average was penalized for playing an extra game against elite competition, while a non-playoff team's metric was calculated strictly over standard regular-season games.
  2. *Franchise relocations and aliases:* Historical franchise moves (St. Louis to Los Angeles, San Diego to Los Angeles, Oakland to Las Vegas) and shifting abbreviation conventions (`JAC`/`JAX`, `WSH`/`WAS`) threatened to drop franchise continuity across years.
  3. *Positional shift gaps:* Relying on `shift(-1)` within a groupby assumes perfectly continuous, unbroken season sequences. If a team or feature row was missing for a single season, `shift(-1)` would silently pair season $t$ with season $t+2$.
* **What We Changed:** In order to resolve all three issues:
  * We explicitly filtered schedules to completed regular-season games (`game_type == 'REG'`).
  * We unified all franchise identifiers across both play-by-play and schedule tables (`TEAM_ALIASES = {'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV', 'JAC': 'JAX', 'WSH': 'WAS'}`).
  * In both the Markov transition table and the supervised dataset, we replaced positional row shifting with explicit calendar year checks: in transitions, we added `.query('season_next == season + 1')`, and in supervised modeling, we explicitly merged on `target_season = season + 1` with `validate='one_to_one'`.

---

### 7. Multi-Algorithm Clustering and Statistical Significance Testing

* **Initial Plan:** Run standard K-Means, pick a $k$, and proceed to modeling.
* **What We Noticed:** Clustering can easily partition random Gaussian noise into visually pleasing clusters. We needed to prove whether these tactical archetypes represented true geometric structure or arbitrary divisions of a continuous space.
* **What We Changed:** 
  1. We benchmarked three distinct clustering families across $k=2\dots 6$: K-Means, Ward Agglomerative Hierarchical, and Gaussian Mixture Models (GMM), tracking Silhouette, Davies-Bouldin, and GMM BIC.
  2. We plotted a truncated Ward linkage dendrogram to inspect the hierarchical merge distances.
  3. We ran a **500-sample bootstrap stability test**, confirming cluster stability with an Adjusted Rand Index (ARI) of $0.709$ (exceeding the standard $0.70$ stability threshold).
  4. We executed a **1,000-iteration column-permutation null test**: shuffling feature columns independently to destroy correlation structure proved that our empirical silhouette score ($0.1864$) significantly exceeded the 99th percentile of null noise ($0.1300$, $p < 0.01$).

---

### 8. Archetype Index Realignment and Label Verification

* **What We Noticed:** After fixing the scramble dropback calculation in the feature matrix, running `KMeans(n_clusters=3, random_state=42)` produced a slight index permutation in the resulting cluster labels: Cluster 0 remained the large short-passing cluster ($52.5\%$), but Cluster 1 was assigned to the high-tempo shotgun cluster ($8.1\%$), and Cluster 2 was assigned to the deep vertical passing cluster ($39.5\%$).
* **What We Changed:** We verified the coordinates of the new cluster centroids and updated the dictionary mapping to ensure descriptive accuracy:
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

* **Initial Plan:** In our first working draft of the notebook, we fitted a single `KMeans(n_clusters=3)` on all 608 team-seasons (2006–2024), computed Euclidean distances to those three centroids, and then passed those pre-computed distance columns into our expanding-window cross-validation loop.
* **What We Noticed:** When critically evaluating our validation design against machine learning best practices, we recognized that this constituted **unsupervised lookahead leakage**: when evaluating our models on 2011–2013 data, the cluster centroids had been fitted on data that included the 2020s. The test folds were being projected onto geometric centers informed by future football.
* **What We Changed:** In order to make our cross-validation strictly leak-free, we moved the unsupervised fitting inside each expanding CV fold:
  ```python
  km_fold = KMeans(n_clusters=3, init='k-means++', n_init=100, random_state=42)
  km_fold.fit(tr[Z_FEATS])
  tr[style_cols] = km_fold.transform(tr[Z_FEATS])
  te[style_cols] = km_fold.transform(te[Z_FEATS])
  ```
  Centroids are now learned exclusively on historical training seasons ($\le t$), and test seasons are transformed using only those historical coordinates.

---

### 10. The Play-Caller vs. Player Dilemma and QB Continuity

* **Context & Motivation:** During our research exploration, we analyzed specific coaching tenures (such as Brian Schottenheimer across NYJ, STL, SEA, and DAL) to evaluate whether offensive coordinators drive offensive success or whether player talent does the heavy lifting. This case study crystallized a central domain insight: an offensive coordinator's performance swings violently when transitioning from an elite quarterback (e.g. Russell Wilson or Dak Prescott) to a backup or developing passer (e.g. Mark Sanchez or Kellen Clemens).
* **What We Noticed:** When building supervised models to test whether a team's offensive style predicts next year's point differential, failure to control for quarterback stability would confound schematic quality with quarterback injuries or departures. Furthermore:
  1. Merging on string names like `passer_player_name` caused potential collisions across players sharing initials and surnames.
  2. At the end of season $t$, an analyst knows a team's offensive style from season $t$ before season $t+1$ starts. However, you cannot know with certainty who will be the *realized* primary starting passer in season $t+1$ (due to unexpected preseason injuries or depth chart changes). Realized QB continuity is technically an **ex-post retrospective control**, not an ex-ante forecast feature.
* **What We Changed:** 
  * We identified each team's primary passer in the neutral sample using unique `passer_player_id` keys.
  * We separated our supervised evaluation into two distinct benchmarks:
    1. **Pure Forecasting Comparison (Ex-ante inputs only):** Comparing past performance ($Y_t$) against past performance plus style ($Y_t$ + Style). Adding style worsened Ridge RMSE from $5.581$ to $5.666$ ($R^2$ dropped from $0.169$ to $0.145$). Tactical style does not improve forward-looking forecasts.
    2. **Retrospective Explanatory Comparison (Controlling for realized QB health):** Comparing $Y_t$ + realized QB continuity against that baseline plus style. Realized QB continuity noticeably improved accuracy (Ridge RMSE dropped to $5.552$, $R^2 = 0.179$), but adding style once again degraded accuracy (Ridge RMSE rose to $5.634$).
  * *Fold 4 Boundary Correction:* We fixed Fold 4 `eval_end` from 2024 to 2023 because the 2025 season has not yet occurred to evaluate a 2024 target.

---

### 11. Fixing the GLS Indexing Bug and Modeling Strategy Decay

* **Initial Plan:** Regress each archetype's annual win premium on its league-wide adoption rate to test if copycat adoption erodes competitive advantage.
* **What We Noticed:** In our initial implementation, `statsmodels.regression.linear_model.GLSAR` threw a runtime `KeyError: 1` when extracting parameters. `sm.add_constant()` produces a pandas Series labeled with column names (`'const'`, `'adoption_rate'`), so positional integer indexing `gls.params[1]` crashed. Furthermore, Ordinary Least Squares (OLS) on stacked time-series panels violates independence assumptions due to serial autocorrelation.
* **What We Changed:** 
  * We fixed parameter extraction using label-safe and positional lookups (`gls.params.iloc[1]`).
  * We checked for autocorrelation using the **Durbin-Watson statistic** and estimated the regression using **Generalized Least Squares with AR(1) errors (GLSAR)**.
  * The resulting slope was negative and statistically significant:
    $$\beta_1 = -5.467, \quad p = 0.013, \quad 95\%\text{ CI: } [-9.733, -1.202]$$
  This empirically verified our Strategy Decay hypothesis: as the league-wide adoption of an offensive archetype expands, its performance premium systematically compresses toward zero as defensive coordinators design schematic countermeasures.

---

### 12. Embracing and Interpreting the Null Hypothesis

* **Initial Expectation:** Like many machine learning projects, there is a natural temptation to hope that newly engineered features (in our case, offensive scheme archetypes) will dramatically improve predictive performance over simple baselines.
* **What the Data Taught Us:** Across Ridge Regression, Random Forest, and LightGBM, adding tactical scheme features consistently worsened out-of-sample error compared to past performance and quarterback stability alone.
* **Our Methodological Stance:** Rather than torturing the data, cherry-picking random seeds, or engineering arbitrary non-linear interactions to force an artificial $R^2$ bump, we **explicitly accepted the null hypothesis**. In the NFL, offensive scheme is an operational conduit through which roster talent and quarterback execution are expressed, not an independent generator of sustainable competitive advantage. Documenting why scheme *fails* to predict future performance—especially alongside our finding that winning schemes decay as they are copied—represents a far more rigorous, honest, and impactful contribution to sports machine learning.

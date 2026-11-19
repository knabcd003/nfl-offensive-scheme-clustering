# Project Evolution and Methodology Notes

**Project:** NFL Offensive Scheme Clustering and Strategic Decay  
**Course:** Principles of Machine Learning I — Fall 2026  
**Authors:** Karthik N., Aneek S., et al.  

---

## Overview

When we initially conceived this project in our project proposal, we had a high-level roadmap: take 23 years of NFL play-by-play data, cluster teams into tactical offensive schemes, and predict future team success to see if offensive scheme serves as a durable competitive advantage.

However, as we moved from abstract proposal to empirical implementation, working directly with raw micro-level event data revealed several critical domain-specific realities and modeling nuances. Each time we dug deeper into the data or inspected model behavior, we adjusted our pipeline to ensure methodological integrity. 

This document tracks how our thinking, data processing, and modeling architecture evolved from our initial proposal to our final notebook deliverable.

---

### 1. Dataset Scope and the 2006 Tracking Boundary

* **Initial Plan:** We originally intended to evaluate all 23 seasons from 2002 to 2024 (736 potential team-seasons across 32 franchises).
* **What We Noticed:** When we audited missingness across individual charting fields in the raw `nflreadpy` logs, we discovered that intended target depth (`air_yards`) and pass directional charting (`pass_location`) were $100\%$ unrecorded from 2002 through 2005. Standard optical tracking and standardized field-location charting only began across all 32 NFL stadiums in **2006**.
* **What We Changed:** In our early exploratory scripts, dropping rows with missing values resulted in 128 "missing" team-seasons ($4\text{ seasons} \times 32\text{ teams} = 128$). Rather than attempting artificial imputation or falling back to crude, non-depth proxies for the early 2000s, we anchored our tactical clustering to **2006–2024** ($19$ completed seasons, $608$ team-seasons). This ensured that every single team-season in our clustering space had complete, uniform tracking data across all spatial dimensions.

---

### 2. Isolating Strategic Intent from Macroeconomic Secular Drift

* **Initial Plan:** We initially considered clustering on absolute frequencies (e.g., raw pass rate, raw shotgun rate, raw no-huddle rate).
* **What We Noticed:** Plotting league-wide historical trends revealed massive macroeconomic drift across eras: neutral-situation shotgun snap rates surged from under $10\%$ in 2002 to nearly $70\%$ by 2024, and pass rates steadily climbed from $54\%$ to over $60\%$. If we clustered on raw rates, an unsupervised algorithm would simply cluster on *time periods*—grouping early-2000s offenses together and 2020s offenses together—rather than identifying genuine tactical philosophy.
* **What We Changed:** In order to decouple organizational style from secular league-wide evolution, we implemented **within-season Z-score standardization**:
  $$Z_{i,t} = \frac{x_{i,t} - \mu_{\text{season}, t}}{\sigma_{\text{season}, t}}$$
  In this standardized space, a positive value always indicates that a team utilized a tactical concept *more than its contemporary peers that specific season*. A $50\%$ shotgun rate in 2003 represented a revolutionary spread pioneer ($+2.8\sigma$), whereas a $50\%$ shotgun rate in 2023 represented a retro, run-heavy offense ($-1.9\sigma$). This transformation allowed us to track the evolution of schematic archetypes across eras on a common, relative scale.

---

### 3. Pure Non-Outcome Features vs. Efficiency Conflation

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

### 4. Accounting for QB Scrambles as Called Dropbacks

* **Initial Plan:** In our first pass at aggregating team play-calling, we calculated pass rate using `play_type == 'pass'`.
* **What We Noticed:** In official NFL box score data and standard play-by-play logs, quarterback scrambles are recorded with `play_type == 'run'`. By filtering strictly on `play_type == 'pass'`, we were inadvertently counting all 9,448 QB scrambles in our neutral sample as designed running plays. Teams with highly mobile quarterbacks (like Michael Vick, Cam Newton, Lamar Jackson, or Josh Allen) were artificially skewed toward appearing like run-heavy ground schemes, even though the coordinator had called a pass play that broke down into a scramble.
* **What We Changed:** In order to measure actual play-calling intent, we reclassified all plays where `qb_scramble == 1` as dropbacks (`qb_dropback = 1`) and defined our pass-rate feature as `grp['qb_dropback'].fillna(0).mean()`. This properly attributed intended passing plays to passing volume regardless of whether the play ended in a throw, a sack, or a scramble.

---

### 5. Regular-Season Target Normalization, Franchise Aliases, and Exact Calendar Joins

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

### 6. Preventing Unsupervised Data Leakage in Cross-Validation

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

### 7. Clarifying Forecasting vs. Retrospective QB Continuity and Unique Player IDs

* **Initial Plan:** In our initial proposal, we identified quarterback stability as a necessary control variable to include alongside offensive style when predicting next season's performance ($Y_{t+1}$), and identified QBs by their displayed text names.
* **What We Noticed:** We identified two important methodological considerations:
  1. *Forecasting vs. Retrospective:* At the end of season $t$, an analyst knows a team's offensive style from season $t$ before season $t+1$ starts. However, you cannot know with certainty who will be the *realized* primary starting passer in season $t+1$ (due to unexpected preseason injuries, depth chart changes, or midseason trades). Realized QB continuity is technically an **ex-post retrospective control**, not an ex-ante forecast feature.
  2. *Player name collisions:* Merging passers on string names like `passer_player_name` (e.g. "J.Allen") can cause collisions across multiple players sharing initials and surnames.
* **What We Changed:** In order to be methodologically transparent and data-safe:
  * We switched from text names to `passer_player_id` to guarantee unique player tracking across seasons.
  * We separated our supervised evaluation into two distinct benchmarks:
    1. **Pure Forecasting Comparison (Ex-ante inputs only):**
       * We compared a forecast baseline ($Y_t$ alone) against a style-augmented forecast ($Y_t$ + fold-specific style distances).
       * *Finding:* Adding style distances raised out-of-sample RMSE (worsening Ridge from $5.581$ to $5.666$) and lowered $R^2$ ($0.169$ to $0.145$). Tactical style does not improve forward-looking forecasts.
    2. **Retrospective Explanatory Comparison (Controlling for realized QB health):**
       * We compared a retrospective baseline ($Y_t$ + realized QB continuity) against that baseline plus style.
       * *Finding:* Controlling for realized QB continuity noticeably improved predictions (Ridge RMSE dropped to $5.552$, $R^2$ rose to $0.179$), but adding style distances once again degraded accuracy (Ridge RMSE rose to $5.634$).
  * *Fold 4 Boundary Correction:* In our initial fold setup, Fold 4 had `eval_end: 2024`. Because 2025 has not yet occurred, 2024 rows had no target $y_{t+1}$ and were dropped. We properly bounded Fold 4 to `eval_end: 2023` (whose target season is 2024).

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

### 8. Econometric Rigor in Testing Strategy Decay

* **Initial Plan:** We hypothesized that winning offensive schemes lose their edge over time due to copycat adoption across the league.
* **What We Noticed:** In testing this relationship, we had pooled cluster-season observations across 19 seasons. Ordinary Least Squares (OLS) regression on time-series panels frequently suffers from serial correlation in the error terms, which can artificially inflate $t$-statistics and produce spurious significance.
* **What We Changed:** In order to test the hypothesis with econometric rigor, we first inspected residual autocorrelation using the **Durbin-Watson statistic** and then estimated the relationship using **Generalized Least Squares with AR(1) errors (GLSAR)**. The resulting slope was negative and statistically significant:
  $$\beta_1 = -5.467, \quad p = 0.013, \quad 95\%\text{ CI: } [-9.733, -1.202]$$
  This provided empirical confirmation of our strategy decay hypothesis: as the league-wide adoption of an offensive archetype expands, its performance premium systematically compresses toward zero as defensive coordinators design schematic countermeasures.

---

### 9. Embracing and Interpreting the Null Hypothesis

* **Initial Expectation:** Like many sports analytics projects, there is a natural temptation to hope that your newly engineered machine learning features (in our case, offensive scheme archetypes) will dramatically improve predictive performance over simple baselines.
* **What the Data Taught Us:** Across Ridge Regression, Random Forest, and LightGBM, adding tactical scheme features consistently worsened out-of-sample error compared to past performance and quarterback stability alone.
* **Our Methodological Stance:** Rather than torturing the data or over-engineering non-linear interactions to force an artificial $R^2$ bump, we **explicitly accepted the null hypothesis**. In the NFL, offensive scheme is an operational conduit through which roster talent and quarterback execution are expressed, not an independent generator of sustainable competitive advantage. Documenting why scheme *fails* to predict future performance—especially alongside our finding that winning schemes decay as they are copied—represents a far more rigorous, honest, and impactful contribution to sports machine learning.

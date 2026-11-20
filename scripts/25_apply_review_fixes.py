import json

def to_nb_lines(text):
    lines = text.splitlines(keepends=True)
    return lines

def main():
    with open('nfl_offensive_clustering.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 48 (Tuning Result)
    cell48_text = """**Tuning Result:**
1. **Adding style continues to raise out-of-sample RMSE across all three models in both benchmarks:**
   - In the *forecasting* comparison, adding style worsens RMSE by $+0.071$ for Ridge, $+0.058$ for Random Forest, and $+0.173$ for LightGBM.
   - In the *retrospective* comparison (with realized QB continuity), adding style worsens RMSE by $+0.070$ for Ridge, $+0.074$ for Random Forest, and $+0.155$ for LightGBM.
   - This confirms that our main conclusion—offensive style provides no marginal predictive utility over past performance—**holds across the hyperparameter settings we tested**, rather than being an artifact of hand-set defaults.
2. **Impact of Tuning:** Inner-CV tuning slightly improved Random Forest (reducing RMSE by $-0.04$ to $-0.06$) and the LightGBM baseline ($-0.033$), while Ridge performance remained within 0.02 points.
3. **Selected Settings:** Across folds, Ridge selected $\\alpha = 10$ in three of four folds (shifting to $\\alpha = 0.1$ in Fold 4), Random Forest favored moderate depths ($2–5$) with conservative leaf sizes ($15–30$), and LightGBM usually preferred shallow leaves ($4$) and heavy regularization ($\\lambda = 15.0$). This indicates that strong regularization was usually preferred on these historical NFL panels."""

    nb['cells'][48]['source'] = to_nb_lines(cell48_text)

    # 2. Update Cell 51 (What the Errors Show)
    cell51_text = """**What the Errors Show:**
1. **Quarterback Changes Drive the Largest Errors:**
   - When a team maintained the same starting quarterback ($n = 267$), the forecast Mean Absolute Error (MAE) was **$4.28$** points per game.
   - When the starting quarterback changed ($n = 149$), the forecast MAE rose to **$5.21$** points per game—nearly a full point ($+0.92$ pts/g, or $21.6\%$) higher.
2. **Zero Systematic Bias Across Archetypes:**
   - The mean residual is near zero across all three offensive archetypes: Deep Vertical Passing ($+0.017$), Short-Depth Passing ($+0.003$), and Shotgun / No-Huddle ($-0.087$).
   - This suggests the forecast model does not systematically favor or penalize any particular offensive system; tactical scheme does not account for the leftover residual variance.
3. **The 10 Largest Misses Are Dominated by Quarterback Transitions:**
   - 7 of the 10 largest forecast misses in the dataset directly involved major quarterback changes or season-ending injuries:
     - *2011 Denver Broncos (missed by $+14.30$ pts):* Tim Tebow departed and Hall of Famer Peyton Manning arrived in 2012 (net points surged from $-5.1$ to $+12.0$).
     - *2016 Los Angeles Rams (missed by $+13.94$ pts):* Sean McVay took over in 2017 with Jared Goff entering year 2 (net points jumped from $-10.6$ to $+9.3$).
     - *2012 Kansas City Chiefs (missed by $+13.90$ pts):* Andy Reid arrived and acquired Alex Smith, jumping from 2–14 to 11–5 ($+7.8$ net points).
     - *2018 San Francisco 49ers ($+12.98$ pts) and 2020 Dallas Cowboys ($+12.22$ pts):* Jimmy Garoppolo and Dak Prescott returned healthy from catastrophic season-ending injuries in year $t$.
     - *2018 Baltimore Ravens ($+12.91$ pts):* Lamar Jackson took over as full-time starter in 2019 and won unanimous MVP.
     - *2023 Dallas Cowboys (missed by $-11.86$ pts):* Dak Prescott suffered a season-ending injury in 2024, causing Dallas to collapse from $+11.4$ to $-6.9$ net points.
   - Unforeseen roster shocks and quarterback health overwhelm offensive scheme as the true determinants of year-over-year NFL performance."""

    nb['cells'][51]['source'] = to_nb_lines(cell51_text)

    # 3. Update Cell 52 (Section 9 Conclusion)
    cell52_text = """## 9. Conclusion, Key Takeaways, and Limitations

### Primary Findings
1. **Three Descriptive Archetypes:** K-Means identifies Short-Depth Passing ($52.5\%$), Shotgun / No-Huddle ($8.1\%$), and Deep Vertical Passing ($39.5\%$). Across $500$ bootstrap resamples, mean ARI was $0.714$ and the 10th percentile was $0.479$. The empirical silhouette of $0.1865$ exceeded the 99th percentile ($0.1306$) of $1,000$ column-permuted samples.
2. **Adoption and Performance Premium:** Higher archetype adoption was associated with a lower average regular-season net points premium in the fitted GLS model ($\\beta = -5.363$, $p = 0.0152$, 95% CI $[-9.650, -1.076]$).
3. **Next-Season Performance:** For the tested models and folds, style distances did not improve forecast scores over past performance alone (and this held after tuning). In the retrospective comparison, style distances also did not improve scores over past performance plus realized QB continuity.
4. **Error Analysis and QB Continuity:** Seasons where the primary starting quarterback changed had about $22\%$ higher out-of-sample forecast error (MAE $5.21$ vs. $4.28$ points/game) under the baseline Ridge model, and $7$ of the $10$ largest prediction misses directly involved starting quarterback injuries or transitions.

### Methodological Limitations
- **Structural Tracking Boundary:** Passing depth (`air_yards`) and directional charting are largely missing before 2006. Clustering uses 2006–2024.
- **Continuous vs. Categorical Scheme Reality:** The modest silhouette ($\\approx 0.19$) indicates overlap among the three tactical groupings. The forecast comparison covers these seven style features summarized by centroid distances.
- **Roster and Coaching:** Overall roster talent is not directly measured. QB continuity is a retrospective proxy based on the most frequent passer in the neutral sample. Coordinator turnover and other personnel changes are unmeasured.
- **Strategy Decay:** The adoption analysis is observational. Its AR(1) specification follows the stacked cluster-season rows, so inference is sensitive to that ordering and to dependence within seasons and archetypes. Each archetype also occupies its own adoption range (the tempo archetype is always rare), so the pooled slope largely reflects differences between archetypes rather than one archetype losing its edge as it spreads."""

    nb['cells'][52]['source'] = to_nb_lines(cell52_text)

    with open('nfl_offensive_clustering.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Successfully updated notebook cells 48, 51, and 52 with clean line formatting.")

if __name__ == '__main__':
    main()

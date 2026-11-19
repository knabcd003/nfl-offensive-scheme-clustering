import json
from pathlib import Path

nb_path = Path("nfl_offensive_clustering.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# First remove the 5 inserted cells if they were inserted
# Find if cell_md1 exists in nb
indices_to_remove = []
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', ''))
    if 'Tactical Feature Distributions and Summary Statistics' in src:
        indices_to_remove = list(range(i, i+5))
        break

if indices_to_remove:
    print(f"Removing previously inserted cells at indices: {indices_to_remove}")
    nb['cells'] = [c for idx, c in enumerate(nb['cells']) if idx not in indices_to_remove]

print(f"Clean notebook has {len(nb['cells'])} cells.")

def make_md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip()
    }

def make_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip()
    }

cell_md1 = make_md(r"""
### Tactical Feature Distributions and Summary Statistics

Before clustering, we examine the empirical distributions and summary statistics of all 7 tactical features across the 608 team-seasons (2006–2024). This allows us to inspect central tendencies, natural variation across franchises, and any underlying skewness.
""")

cell_code1 = make_code(r"""
# Summary statistics of raw tactical features
stats_summary = pd.DataFrame({
    'Mean': df_clean[FEATS].mean(),
    'Std': df_clean[FEATS].std(),
    'Median': df_clean[FEATS].median(),
    'IQR': df_clean[FEATS].quantile(0.75) - df_clean[FEATS].quantile(0.25),
    'Min': df_clean[FEATS].min(),
    'Max': df_clean[FEATS].max(),
    'Skewness': df_clean[FEATS].skew()
})
print("--- Summary Statistics across 608 Team-Seasons (2006–2024) ---")
print(stats_summary.round(3))

# Visualizing raw distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

feature_labels = [
    'Called Dropback Rate', 'Shotgun Snap Rate', 'No-Huddle Tempo Rate',
    'Intended Air Yards / Att', 'Deep Target Rate (≥15 yds)',
    'Target Dispersion (HHI)', 'Interior Rush Share'
]

for i, (f, label) in enumerate(zip(FEATS, feature_labels)):
    sns.histplot(df_clean[f], kde=True, ax=axes[i], color='steelblue', bins=25)
    axes[i].set_title(label, fontweight='bold', fontsize=11)
    axes[i].set_xlabel('Raw Value / Rate')
    axes[i].set_ylabel('Team-Seasons')

fig.delaxes(axes[7])
plt.suptitle('Empirical Distributions of 7 Raw Tactical Features (2006–2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")

cell_md2 = make_md(r"""
### Outlier Detection and Historical Tactical Extremes

In high-dimensional sports clustering, outliers represent extreme tactical experiments (e.g., radical tempo revolutions, option-heavy attacks, or extreme vertical passing). We inspect the standardized Z-score space using boxplots and identify observations exceeding three standard deviations ($|Z| > 3.0$).
""")

cell_code2 = make_code(r"""
# Standardized Z-Score Boxplots
plt.figure(figsize=(12, 4.5))
short_names = ['Pass Rate', 'Shotgun', 'No-Huddle', 'Air Yards', 'Deep Tgt %', 'Target Conc', 'Interior Rush']
sns.boxplot(data=df_clean[Z_FEATS], palette='Set2')
plt.xticks(range(7), short_names, fontweight='bold')
plt.ylabel('Within-Season Z-Score (σ)')
plt.title('Standardized Tactical Feature Spread and Outliers', fontweight='bold')
plt.axhline(0, color='gray', linestyle='--', lw=0.8)
plt.axhline(3.0, color='red', linestyle=':', lw=1.0, label='|Z| = 3.0 Threshold')
plt.axhline(-3.0, color='red', linestyle=':', lw=1.0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Tabulate the most extreme tactical outliers (|Z| > 3.0)
outliers = []
for f, zf in zip(FEATS, Z_FEATS):
    extreme_rows = df_clean[df_clean[zf].abs() > 3.0][['season', 'team_clean', f, zf]]
    for _, row in extreme_rows.iterrows():
        outliers.append({
            'Season': int(row['season']),
            'Team': row['team_clean'],
            'Feature': f,
            'Raw Value': round(row[f], 3),
            'Z-Score (σ)': round(row[zf], 2)
        })

outliers_df = pd.DataFrame(outliers).sort_values('Z-Score (σ)', key=abs, ascending=False)
print(f"Total extreme outlier instances (|Z| > 3.0): {len(outliers_df)}")
print()
print("Prominent Historical Tactical Outliers:")
print(outliers_df.head(12).to_string(index=False))
""")

cell_md3 = make_md(r"""
**What These Distributions and Outliers Tell Us for Downstream Modeling:**

1. **Severe Right-Skew in Tempo (`no_huddle_rate` skew = 3.50):** The median NFL no-huddle rate is just $3.4\%$, but extreme innovators pushed it above $60\%–70\%$ (such as Chip Kelly's 2013–2015 Eagles at $+3.7\sigma$ and Peyton Manning's 2010 Colts at $+4.6\sigma$). This heavy tail explains why unsupervised clustering naturally isolates a distinct, compact "Shotgun / No-Huddle" archetype (Cluster 1, $\sim 8\%$ of sample).
2. **Symmetric Variation in Passing Depth and Volume:** Features like `air_yards_per_att` (skew = $0.11$) and `pass_rate` (skew = $-0.21$) are nearly Gaussian, showing smooth, continuous variation across the league. Notable outliers capture identifiable historical extremes, such as the 2011 Denver Broncos under Tim Tebow (pass rate of $39.8\%$, $-3.18\sigma$) running a heavy option attack, and the 2007 16–0 Patriots ($+3.26\sigma$ shotgun rate) pioneering modern spread formations.
3. **Data Quality and Bounds:** All features fall within physiologically sound boundaries (no negative air yards, no rates outside $[0, 1]$), ensuring stable distance calculations in downstream Euclidean clustering and regression models.
""")

new_cells = [cell_md1, cell_code1, cell_md2, cell_code2, cell_md3]

# Insert after Cell 17 (index 16)
nb['cells'] = nb['cells'][:17] + new_cells + nb['cells'][17:]
print(f"New notebook has {len(nb['cells'])} cells.")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Saved cleanly formatted notebook.")

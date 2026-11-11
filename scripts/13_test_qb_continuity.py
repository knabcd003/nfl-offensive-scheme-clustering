import pandas as pd
import numpy as np

# Can we determine the primary passer per team-season directly from pbp or load_player_stats?
df_pbp = pd.read_parquet("data_cache/pbp_neutral_clean.parquet")

# In df_pbp, we have passer_player_name or passer_player_id
# Let's count passes per passer per team-season
passer_counts = (df_pbp[df_pbp['play_type'] == 'pass']
                 .groupby(['season', 'posteam', 'passer_player_name'])
                 .size()
                 .reset_index(name='pass_attempts'))

# Primary passer = passer with the most attempts for that team that season
idx_max = passer_counts.groupby(['season', 'posteam'])['pass_attempts'].idxmax()
primary_passers = passer_counts.loc[idx_max].rename(columns={'posteam': 'team', 'passer_player_name': 'primary_qb'})
print("Sample primary passers:")
print(primary_passers.head(10))

# Now let's see: for team in season t and t+1, is the primary QB the same?
primary_passers['team_clean'] = primary_passers['team'].replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
primary_passers = primary_passers.sort_values(['team_clean', 'season'])
primary_passers['next_season'] = primary_passers['season'] + 1
primary_passers['next_primary_qb'] = primary_passers.groupby('team_clean')['primary_qb'].shift(-1)
primary_passers['qb_continuity'] = (primary_passers['primary_qb'] == primary_passers['next_primary_qb']).astype(int)

print("\nQB Continuity rate:")
print(primary_passers['qb_continuity'].value_counts(normalize=True))
print("\nSample continuity transitions:")
print(primary_passers[['season', 'team_clean', 'primary_qb', 'next_primary_qb', 'qb_continuity']].head(15))

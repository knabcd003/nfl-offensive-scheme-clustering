import pandas as pd
from pathlib import Path

df_pbp = pd.read_parquet("data_cache/pbp_neutral_clean.parquet")

# Inspect pass plays specifically
passes = df_pbp[df_pbp['play_type'] == 'pass']
runs = df_pbp[df_pbp['play_type'] == 'run']

print("Pass plays per season and air_yards missing rate:")
print(passes.groupby('season')['air_yards'].apply(lambda s: s.isna().mean()))

print("\nPass location missing rate on pass plays:")
print(passes.groupby('season')['pass_location'].apply(lambda s: s.isna().mean()))

print("\nRun location missing rate on run plays:")
print(runs.groupby('season')['run_location'].apply(lambda s: s.isna().mean()))

print("\nShotgun missing rate:")
print(df_pbp.groupby('season')['shotgun'].apply(lambda s: s.isna().mean()))

print("\nNo-huddle missing rate:")
print(df_pbp.groupby('season')['no_huddle'].apply(lambda s: s.isna().mean()))

import pandas as pd
import numpy as np

df_pbp = pd.read_parquet("data_cache/pbp_neutral_clean.parquet")

# Check down column
print("Neutral plays by down:")
print(df_pbp['down'].value_counts(dropna=False).sort_index())

# Check receiver position columns
rec_cols = [c for c in df_pbp.columns if 'receiver' in c or 'position' in c]
print("\nReceiver columns:", rec_cols)

# Check pass_location values
print("\nPass locations (2006+):")
print(df_pbp[df_pbp['season'] >= 2006]['pass_location'].value_counts(dropna=False))

# Check run_location values
print("\nRun locations (2006+):")
print(df_pbp[df_pbp['season'] >= 2006]['run_location'].value_counts(dropna=False))


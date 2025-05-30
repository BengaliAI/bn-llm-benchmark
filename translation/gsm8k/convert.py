import pandas as pd

# Load the Parquet file
df = pd.read_parquet('train-00000-of-00001.parquet')

# Save the DataFrame as JSON Lines format
df.to_json('gsm8k_main_train.jsonl', orient='records', lines=True)
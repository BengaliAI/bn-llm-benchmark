import pandas as pd

# Load the Parquet file
df = pd.read_parquet('test-00000-of-00001.parquet')

# Save the DataFrame as JSON Lines format
df.to_json('MMLU_PRO_test.jsonl', orient='records', lines=True)

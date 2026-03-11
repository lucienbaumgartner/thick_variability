import pandas as pd
from collections import defaultdict
import random

# Load CSV
df = pd.read_csv("../../output/corpora/corpus.csv")

print(df.columns.values)

df['word_count'] = df['sentence'].str.findall(r'\w+').str.len()
mean_wc = df['word_count'].mean()
std_wc = df['word_count'].std()

lower = mean_wc - std_wc / 4  # ~16.5
upper = mean_wc + std_wc / 4  # ~31.5

df_filtered = df[(df['word_count'] >= lower) & (df['word_count'] <= upper)]

print(f"Mean word count: {mean_wc:.2f}")
print(f"1 standard deviation: {std_wc:.2f}")



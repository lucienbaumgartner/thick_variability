import pandas as pd
import ast
import json

input_file = "../../output/corpora/reddit/built_corpus/corpus_with_duplicates.csv"
df = pd.read_csv(input_file)

# If they are strings, convert to dict safely
def safe_parse(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df["target_dep_"] = df["target_dep_"].apply(safe_parse)
df["target_neg"] = df["target_neg"].apply(safe_parse)

# Convert dicts to a stable string representation
df["target_dep_str"] = df["target_dep_"].apply(lambda d: json.dumps(d, sort_keys=True))
df["target_neg_str"] = df["target_neg"].apply(lambda d: json.dumps(d, sort_keys=True))

# Drop duplicates based on the 4 keys
df_dedup = df.drop_duplicates(
    subset=["sentence", "author", "target_dep_str", "target_neg_str"]
).copy()

# Ensure unique IDs
df_dedup["id"] = (
    df_dedup.groupby("id").cumcount()
    .astype(str)
    .radd(df_dedup["id"] + "_")
    .str.replace("_0", "", regex=False)
)

df_dedup = df_dedup.drop(columns=["target_dep_str", "target_neg_str"])

# Write out
output_file = "../../output/corpora/reddit/built_corpus/deduplicated_corpus.csv"
df_dedup.to_csv(output_file, index=False)
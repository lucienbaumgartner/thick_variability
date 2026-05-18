import subprocess
import os
import glob
import json

dir_path = "workers/term_filtering"

if os.path.exists(os.path.join(dir_path, "status.json")):
    for file_path in glob.glob(os.path.join(dir_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

dir_path = "../../output/corpora/reddit/term_filtering"
for file_path in glob.glob(os.path.join(dir_path, "*")):
    if os.path.isfile(file_path):
        os.remove(file_path)

tt_path = "../../input/target_terms.json"
with open(tt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_terms = []
for item in data:
    all_terms.extend(item["pos"])
    all_terms.extend(item["neg"])

target_string = ",".join(all_terms)

print(target_string)

# Define the command as a list of arguments
command = [
    "python3", "helper/combine_folder_multiprocess.py",
    "../../output/corpora/reddit/subreddit_filtering",
    "--working", "workers/term_filtering",
    "--field", "selftext",
    "--value", target_string,
    "--partial",
    "--output", "../../output/corpora/reddit/term_filtering"
]

# Run the command
subprocess.run(command, check=True)

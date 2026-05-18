import subprocess
import os
import glob
import pathlib

dir_path = "workers/subreddit_filtering"

if os.path.exists(os.path.join(dir_path, "status.json")):
    for file_path in glob.glob(os.path.join(dir_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

dir_path = "../../output/corpora/reddit/subreddit_filtering"
for file_path in glob.glob(os.path.join(dir_path, "*")):
    if os.path.isfile(file_path):
        os.remove(file_path)

# Define the command as a list of arguments
command = [
    "python3", "helper/combine_folder_multiprocess.py",
    "../../../data/reddit/submissions",
    "--working", "workers/subreddit_filtering",
    "--field", "subreddit",
    "--value", "relationships,relationship_advice,AmItheAsshole,confession,offmychest,CasualConversation,MadeMeSmile,HumansBeingBros,GetMotivated,wholesome,WritingPrompts",
    "--processes", "20",
    "--output", "../../output/corpora/reddit/subreddit_filtering",
]

# Run the command
subprocess.run(command, check=True)

from pathlib import Path

dir_path = Path("../../output/corpora/reddit/subreddit_filtering")
prefix = "RS_"

for file_path in dir_path.iterdir():
    if file_path.is_file():  # skip directories
        new_name = prefix + file_path.name
        new_path = file_path.with_name(new_name)
        file_path.rename(new_path)
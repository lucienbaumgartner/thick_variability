import subprocess

# Define the command as a list of arguments
command = [
    "python3", "combine_folder_multiprocess.py",
    "../../../data/reddit/submissions",
    "--field", "selftext",
    "--value", "cruel,compassionate,rude,friendly,manipulative,honest,selfish,generous,wrong,right,good,bad",
    "--partial",
    "--output", "../../output/corpora"
]

# Run the command
subprocess.run(command, check=True)
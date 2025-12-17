import os
import re

# Path to your directory
folder_path = "raw_audio"

# Get all wav files
files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Function to extract number from filename
def extract_number(filename):
    return int(re.search(r'\d+', filename).group())

# Sort files by numeric value
files.sort(key=extract_number)

# Rename files
for idx, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_name = f"audio_{idx}.wav"
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)

print("Renaming completed successfully!")

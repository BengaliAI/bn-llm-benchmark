#!/usr/bin/env python3

import os
import shutil

# Get the current working directory
cwd = os.getcwd()

# Iterate over all entries in the current directory
for filename in os.listdir(cwd):
    # Check if the entry is a file and ends with .json
    if filename.endswith('.json') and os.path.isfile(filename):
        # Extract the filename without the .json extension
        dirname = filename[:-5]  # Remove the last 5 characters (.json)
        # Create a directory with the extracted name (if it doesn't already exist)
        os.makedirs(dirname, exist_ok=True)
        # Move the .json file into the newly created directory
        shutil.move(filename, os.path.join(dirname, filename + "l"))

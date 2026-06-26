#!/bin/bash

# 1. Check if the required environment variable is set
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set."
    exit 1
fi

# 2. Configuration
SAVE_DIR="$NAVSIM_DEVKIT_ROOT/models"
# Google Drive Folder ID
FOLDER_ID="1kxL20MzLynbNuKvMzT64gcXsWxxIcK0y"

# Create the save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# 3. Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing via pip..."
    pip install gdown
fi

# 4. Download the entire folder
echo "Downloading all files from the Google Drive folder..."
# The --folder option downloads all contents within the specified folder ID
gdown --folder "$FOLDER_ID" -O "$SAVE_DIR"

# 5. Verify the download
if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "Success: All files have been saved to $SAVE_DIR"
    echo "Directory contents:"
    ls -lh "$SAVE_DIR"
else
    echo "Download failed. Please check the folder link permissions or your network connection."
    exit 1
fi
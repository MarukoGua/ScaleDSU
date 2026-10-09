#!/bin/bash

CURRENT_SCRIPT=$(basename "$0")
TARGET_DIR="./scripts"

echo "$CURRENT_SCRIPT"

for file in "$TARGET_DIR"/*; do
    if [ -f "$file" ] && [ "${file##*.}" = "sh" ]; then

        file_name=$(basename "$file")

        echo "--> $file_name"
        
        if [ "$file_name" = "$CURRENT_SCRIPT" ]; then
            continue
        fi
        
        chmod +x "$file"
        
        "$file"
        
        echo "--> Finish: $file"
        echo "--------------------------------------------------"
    fi
done
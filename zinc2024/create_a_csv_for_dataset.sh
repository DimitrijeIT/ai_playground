#!/bin/bash

# Check if a directory has been provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

directory=$1
output_file="output.csv"

# Write CSV header
echo "file_name,label" > "$output_file"

# Iterate over files in the directory
for file in "$directory"/*; do
  if [ -f "$file" ]; then
    # Extract the file name and the 4-digit label
    file_name=$(basename "$file")
    label=$(echo "$file_name" | grep -oP 'p\K\d{4}')

    # Write to CSV
    echo "$file_name,$label" >> "$output_file"
  fi
done

echo "CSV file created: $output_file"

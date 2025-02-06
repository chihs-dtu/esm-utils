#!/bin/bash

INPUT_FILE="example_proteins/example_proteins.fasta"
DIR_NAME="example_proteins/"
# Check for example files
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Example protein FASTA file not found."
  exit 1
fi

# Run the script
conda activate esm2
if python run.py -n $INPUT_FILE; then
	# Check the output files
	for filename in esm2enc attention; do
		if [ -f "${DIR_NAME}/${filename}.json" ]; then
			# Check the dimension
			if python -c '
import json
obj = json.load(open("${DIR_NAME}/${filename}}.json","r"))
if len(obj)!=5: 
	exit(1)
else 
	exit(0)
';
			then
				echo "Error: The output pickle file has incorrect dimension."
				exit 1
			fi
		fi
	done
	echo "Success!"
else
	echo "Error: Failed to run the script."
fi

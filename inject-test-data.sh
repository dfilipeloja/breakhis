#!/bin/bash

DIRBENIGN=./train/benign
DIRMALIGNANT=./train/malignant

for file in $(cat ./benign-imgs.txt)
do
	FILE=$DIRBENIGN/$file

	if [[ -f "$FILE" ]]; then
		mv "$FILE" ./test/benign/;
	fi
done

for file in $(cat ./malignant-imgs.txt)
do
	FILE=$DIRMALIGNANT/$file
	
	if [[ -f "$FILE" ]]; then
		mv "$FILE" ./test/malignant/;
	fi
done

echo "Images copied to their respective directories..."

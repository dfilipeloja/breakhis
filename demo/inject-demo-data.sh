#!/bin/bash

DIRBENIGN=../test/benign
DIRMALIGNANT=../test/malignant

for file in $(cat ./benign-demo.txt)
do
	FILE=$DIRBENIGN/$file

	if [[ -f "$FILE" ]]; then
		cp "$FILE" ./benign/;
	fi
done

for file in $(cat ./malignant-demo.txt)
do
	FILE=$DIRMALIGNANT/$file
	
	if [[ -f "$FILE" ]]; then
		cp "$FILE" ./malignant/;
	fi
done

echo "Images copied to their respective directories..."

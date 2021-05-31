#!/bin/bash
for file in $(cat benign_images_test.txt)
do
	FILE=./train/benign/$file

	if [[ -f "$FILE_B" ]]; then
		mv "$FILE_B" ./test/benign/;
	fi
done

for file in $(cat malignant_images_test.txt)
do
	FILE=./train/malignant/$file
	
	if [[ -f "$FILE" ]]; then
		mv "$FILE" ./test/malignant/;
	fi
done

echo "Images moved to their respective directories..."

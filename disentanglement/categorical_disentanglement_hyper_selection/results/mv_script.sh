#!/bin/sh

# Get all directory names in the current directory
directories=$(find . -maxdepth 1 -type d | sed 's|^\./||')

# Iterate over the directory names
for ARGUMENT in $directories
do
        mv $ARGUMENT/train_losses.csv $ARGUMENT/${ARGUMENT}_train_losses.csv
done

ldconfig

echo "Done!"


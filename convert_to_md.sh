#!/bin/bash

# Check if the notebook file exists
if [ ! -f "RE-GPT.ipynb" ]; then
    echo "Nootebook file not found!"
    exit 1
fi

# Convert the Jupyter notebook to a markdown file and embed images
jupyter nbconvert --to markdown RE-GPT.ipynb --NbConvertApp.output_base=README --embed-images

# Check if the conversion was successful
if [ $? -eq 0 ]; then
    echo "Notebook successfully converted to README.md with embedded images."
else
    echo "Conversion failed."
    exit 1
fi

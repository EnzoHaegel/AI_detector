#!/bin/bash

ENV_NAME="gptzero_env"
PYTHON_VERSION="python3.9"
REQUIREMENTS_FILE="requirements.txt"

echo "==== Initializing environment for GPTZero ===="

if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "Error: $PYTHON_VERSION is not installed. Please install it first."
    exit 1
fi

echo "Creating virtual environment..."
$PYTHON_VERSION -m venv $ENV_NAME

source "$ENV_NAME/Scripts/activate"

echo "Updating pip..."
pip install --upgrade pip

if [ ! -f $REQUIREMENTS_FILE ]; then
    cat <<EOT > $REQUIREMENTS_FILE
transformers
torch
pandas
openpyxl
xlsxwriter
EOT
fi

echo "Installing dependencies..."
pip install -r $REQUIREMENTS_FILE

echo "==== Environment is ready! ===="
echo "To activate the environment, use:"
echo "source $ENV_NAME/Scripts/activate"

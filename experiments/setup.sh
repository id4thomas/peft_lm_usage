#!/bin/bash

# PREPARE ENV
MODEL_SAVE_DIR=""
DATA_DIR=""

if [ ! -d "${MODEL_SAVE_DIR}" ]; then
    mkdir "${MODEL_SAVE_DIR}"
    echo "Folder '${MODEL_SAVE_DIR}' created."
else
    echo "Folder '${MODEL_SAVE_DIR}' already exists."
fi

if [ ! -d "${DATA_DIR}" ]; then
    mkdir "${DATA_DIR}"
    echo "Folder '${DATA_DIR}' created."
else
    echo "Folder '${DATA_DIR}' already exists."
fi

### MAKE VENV
python3.8 -m venv env
source env/bin/activate
pip install -r requirements/requirements.torch2.txt
deactivate
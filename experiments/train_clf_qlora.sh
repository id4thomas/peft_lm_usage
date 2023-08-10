#!/bin/bash
TASK="clf"
MODEL_DIR=""
CONFIG_DIR=""

DATA_DIR=""
DATA_PREFIX=""

source env/bin/activate

echo "python src/seq_clf/train_clf_qlora.py --project_name ${TASK} --config_dir ${CONFIG_DIR} --model_save_dir ${MODEL_DIR} --data_dir=${DATA_DIR} --data_prefix ${DATA_PREFIX}"

python src/seq_clf/train_clf_qlora.py \
	--project_name ${TASK} \
	--config_dir ${CONFIG_DIR} \
	--model_save_dir ${MODEL_DIR} \
	--data_dir=${DATA_DIR} \
	--data_prefix ${DATA_PREFIX}

deactivate
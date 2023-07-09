WORKDIR="/usr/app"

## Train Params
PROJECT_NAME="lora_demo"
CONFIG="${WORKDIR}/experiments/config/lora_demo.json"
MODEL_SAVE_DIR="${WORKDIR}/experiments/models"
DATA_DIR="${WORKDIR}/sample_data/causal_lm"
DATA_PREFIX="causal"

cd ${WORKDIR}/src/causal_lm

source ${WORKDIR}/env/bin/activate
python train_lm_lora.py \
	--project_name ${PROJECT_NAME} \
	--config_dir ${CONFIG} \
	--model_save_dir ${MODEL_SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--data_prefix ${DATA_PREFIX}

deactivate
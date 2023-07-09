import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = ""
import argparse

# Training Frameworks
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTNeoXForSequenceClassification
from transformers import Trainer, TrainingArguments

from utils.common_utils import set_seed
from utils.peft_utils import get_lora_config, get_lora_model, get_lora_save_param_dict

## logging
from utils.wandb_utils import wandb_set, wandb_watch_model

# Etc Utils
import json
import evaluate
import numpy as np

from utils.data_utils import load_data_to_dataset, process_dataset

def load_dataset(data_dir, data_prefix, tokenizer, config):
	train_ds = load_data_to_dataset(os.path.join(data_dir, f"{data_prefix}-train.tsv"))
	val_ds = load_data_to_dataset(os.path.join(data_dir, f"{data_prefix}-train.tsv"))

	# Preprocess
	train_ds = process_dataset(train_ds, tokenizer, config)
	val_ds = process_dataset(train_ds, tokenizer, config)
	return train_ds, val_ds

## Train Preparation
def prepare_lora_model(config):
	pretrained_model_name = config.get("pretrained_model")
	pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels = 2)

	# lora_config
	lora_config = get_lora_config(config)
	lora_model = get_lora_model(pretrained_model, lora_config)

	return lora_model


def compute_metrics(eval_pred):
	metric = evaluate.load("accuracy")
	labels = eval_pred.label_ids
	predictions = eval_pred.predictions.argmax(-1)
	return metric.compute(predictions=predictions, references=labels)

def get_training_args(config, train_run_name, model_save_dir):
	return TrainingArguments(
		run_name = train_run_name,

		# Train Params
		## Steps/Epochs
		num_train_epochs = config["epochs"],
		# max_steps = 2,

		## LR
		learning_rate = config["learning_rate"],
		## Batch
		per_device_train_batch_size = config["per_device_batch_size"],
		per_device_eval_batch_size = config["per_device_batch_size"],
		gradient_accumulation_steps = config["gradient_accumulation_steps"],
		## ETC
		# label_smoothing_factor = config["label_smoothing_factor"],

		# Checkpointing, Saving
		output_dir = os.path.join(model_save_dir, "checkpoints"),
		save_strategy = "steps", # steps, epoch
		save_steps = config["save_steps"],
		save_total_limit = config["save_total_limit"],
		load_best_model_at_end = True,
		overwrite_output_dir=True,

		# Evaluating
		evaluation_strategy = "steps",
		metric_for_best_model = config["metric_for_best_model"],

		# Logging
		logging_dir = model_save_dir,
		logging_steps = config["summary_step"],
		disable_tqdm = False,
		report_to = "wandb",

		# System
		seed = config["seed"],
		fp16 = config["fp16"],
		bf16 = config["bf16"]
	)

def train(config, save_trained = True):
	# random seed
	set_seed(config.get("seed", 42))

	# Load Data
	pretrained_model_name = config.get("pretrained_model")
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

	train_ds, val_ds = load_dataset(
		config.get("data_dir"),
		config.get("data_prefix"),
		tokenizer,
		config
	)
	
	# Load Model
	lora_model = prepare_lora_model(config)

	# Prepare training
	## LOGGING
	per_device_batch_size = config["per_device_batch_size"]
	gradient_accumulation_steps = config["gradient_accumulation_steps"]
	cuda_device_count = torch.cuda.device_count()
	effective_batch_size = per_device_batch_size*gradient_accumulation_steps*cuda_device_count

	
	train_run_name = "{}_lora_r{}_alpha{}_ep{}_batch{}_lr{}".format(
		pretrained_model_name.split("/")[-1], 
		config["lora_r"],
		config["lora_alpha"],
		config["epochs"],
		effective_batch_size,
		config["learning_rate"]
	)

	wandb_set(
		config.get("project_name"),
		config.get("config_dir"),
		train_run_name
	)

	wandb_watch_model(
		lora_model, 
		log_freq = config.get("summary_step", 10)
	)
	## SAVING
	model_save_dir = os.path.join(config.get("model_save_dir"), train_run_name)

	## Training Arguments
	training_args = get_training_args(config, train_run_name, model_save_dir)

	# Train
	trainer = Trainer(
		model = lora_model,
		args = training_args,
		# data_collator=data_collator,
		train_dataset = train_ds,
		eval_dataset = val_ds,
		compute_metrics=compute_metrics
	)

	# Train
	trainer.train()
	trainer.evaluate(val_ds, metric_key_prefix = "final")

	if save_trained:
		best_dir = os.path.join(model_save_dir, "best")
		if not os.path.exists(best_dir):
			os.makedirs(best_dir)

		lora_save_dict = get_lora_save_param_dict(trainer.model, save_embedding = False)
		torch.save(lora_save_dict, os.path.join(best_dir, "model.pt"))

		tokenizer.save_pretrained(str(best_dir))
		with open(os.path.join(best_dir, "train_config.json"), 'w') as f:
			json.dump(config, f, indent = 4, ensure_ascii = False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--project_name', default='lora_lm')
	parser.add_argument('--config_dir', default='config')
	parser.add_argument('--model_save_dir', default='models')
	parser.add_argument('--data_dir', default='data')
	parser.add_argument('--data_prefix', default='data')
	args = parser.parse_args()

	# Load Config
	with open(args.config_dir, 'r') as f:
		config = json.loads(f.read())

	# Add argparse contents to config
	config.update(vars(args))

	train(config, save_trained = True)

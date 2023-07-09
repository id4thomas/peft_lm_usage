import csv
import pandas as pd
from datasets import Dataset

def batch_tokenize_preprocess_dec(
		batch, 
		tokenizer, 
		max_length, 
		input_template = "{} {}", 
		add_eos_token = True
	):
	source, target = batch["source"], batch["target"]
	
	if add_eos_token:
		input_template += tokenizer.eos_token
	input_sents = [input_template.format(s,t) for s,t in zip(source, target)]
	
	tokenized = tokenizer(input_sents, 
								 truncation=True, 
								 max_length=max_length, 
								 padding="max_length", add_special_tokens = True)
	  
	batch = {
		"input_ids": tokenized["input_ids"], 
		"attention_mask": tokenized["attention_mask"]
	}

	batch["source"] = source
	batch["target"] = target
	return batch

def load_data_to_dataset(fpath):
	df = pd.read_csv(fpath, sep = "\t", quoting = csv.QUOTE_NONNUMERIC)
	return Dataset.from_pandas(df)

def process_dataset(dataset, tokenizer, config):
	return dataset.map(
		lambda batch: batch_tokenize_preprocess_dec(
			batch,
			tokenizer,
			config.get("max_length"),
			input_template = config.get("input_template"),
			add_eos_token = True
		),
		batched = True
	)
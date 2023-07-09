import csv
import pandas as pd
from datasets import Dataset

def batch_tokenize_preprocess_enc_clf(
		batch, 
		tokenizer, 
		max_length
	):
	source, labels = batch["source"], batch["labels"]
	
	tokenized = tokenizer(source, 
								 truncation=True, 
								 max_length=max_length, 
								 padding="max_length", add_special_tokens = True)
	  
	batch = {
		"input_ids": tokenized["input_ids"], 
		"attention_mask": tokenized["attention_mask"],
		# "token_type_ids": tokenized["token_type_ids"]
	}

	batch["source"] = source
	batch["labels"] = labels
	return batch

def load_data_to_dataset(fpath):
	df = pd.read_csv(fpath, sep = "\t")
	return Dataset.from_pandas(df)

def process_dataset(dataset, tokenizer, config):
	return dataset.map(
		lambda batch: batch_tokenize_preprocess_enc_clf(
			batch,
			tokenizer,
			config.get("max_length")
		),
		batched = True
	)
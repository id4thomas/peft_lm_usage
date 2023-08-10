from peft import get_peft_model, TaskType, get_peft_model_state_dict

# Lora Utils
from peft import LoraConfig
from peft import PrefixTuningConfig

def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)

def get_pretrained_model_for_kbit_training(model):
	try:
		from peft import prepare_model_for_kbit_training
	except ImportError as error:
		raise NotImplementedError("prepare_model_for_kbit_training is available for peft>=0.4.0")
	except Exception as e:
		raise e

	model.gradient_checkpointing_enable()
	model = prepare_model_for_kbit_training(model)
	return model

## Lora Utils
def get_lora_config(config):
	return LoraConfig(
			task_type=TaskType.SEQ_CLS, 
			inference_mode=False, 
			r = config["lora_r"], 
			lora_alpha = config["lora_alpha"], 
			lora_dropout = config["lora_dropout"]
		)

def get_lora_model(causal_lm, peft_config):
	return get_peft_model(causal_lm, peft_config)

def get_lora_save_param_dict(model, save_embedding = False):
	state_dict = model.state_dict()
	params_to_save = get_peft_model_state_dict(model, state_dict=state_dict)
	
	if save_embedding:
		layer_keys = list(state_dict.keys())
		embed_keys = list(filter(lambda x: "embed_in" in x, layer_keys))
		for k in embed_keys:
			params_to_save[k] = state_dict[k]
			
	return params_to_save

## Prefix-tuning Utils
def get_prefixtuning_config(config):
	return PrefixTuningConfig(
			task_type=TaskType.CAUSAL_LM, 
			inference_mode=False, 
			num_virtual_tokens = config["num_virtual_tokens"]
	)
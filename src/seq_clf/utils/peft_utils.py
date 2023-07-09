from peft import get_peft_model, TaskType, get_peft_model_state_dict

# Lora Utils
from peft import LoraConfig
from peft import PrefixTuningConfig

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
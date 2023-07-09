# PEFT Usage Samples
* Examples of Huggingface PEFT (Parameter Efficient Fine-tuning) package
    * https://github.com/huggingface/peft - v0.3.0
* Mainly Korean LMs, 주로 한국어 모델 기준 테스트

## Tasks
* Causal LM (src/causal_lm)
    * train_lm_lora - for AutoModelForCausalLM usage
* Sequence Classification (src/seq_clf)
    * train_clf_lora - for AutoModelForSequenceClassification usage
    * train_clf_gptneox_lora - for GPTNeoXForSequenceClassification usage (for custom compute_metrics)

## Testing Environment
* Env1:
    * CPU: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
    * GPU: Titan RTX (Driver 525.105.17)
    * CUDA 11.7 based Docker img
```
torch==2.0.1
transformers==4.30.2
peft==0.3.0
```

* Env2: Macbook Pro M1 Max 32GB
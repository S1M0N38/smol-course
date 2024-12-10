########################################################################################
# %% Imports
########################################################################################

import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

########################################################################################
# %% Load Dataset
########################################################################################

assert load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
assert DATA_PATH, "Please set the DATA_PATH environment variable"
data_path = Path(DATA_PATH)

dataset = load_dataset(path="trl-lib/ultrafeedback_binarized")
print(dataset)
print(dataset["train"][0])  # type: ignore

########################################################################################
# %% Load Base Model
########################################################################################

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = setup_chat_format(model, tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = f"smol-course-2.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
finetune_tags = ["smol-course", "module_2"]


########################################################################################
# %% Odds Ratio Preference Optimization (ORPO) with TRL
########################################################################################

orpo_config = ORPOConfig(
    num_train_epochs=1,
    # Lower values (like 0.1) make the model more conservative in following preferences
    beta=0.1,
    # Optimize traning for low-end hardware
    per_device_train_batch_size=8,
    # gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    # eval_accumulation_steps=1,
    optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
    # Learning rate
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    warmup_steps=100,
    # Dataset
    remove_unused_columns=True,
    # Tokens
    max_prompt_length=512,  # input tokens
    max_length=1024,  # input tokens + output tokens
    # Attributes for saving and loading
    output_dir=str(data_path / finetune_name),
    hub_model_id=finetune_name,
    run_name=finetune_name,
    # Logging and Eval
    eval_strategy="steps",
    eval_steps=0.1,  # fractional epochs
    logging_steps=16,
    report_to="wandb",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset["train"],  # type: ignore
    eval_dataset=dataset["test"],  # type: ignore
    processing_class=tokenizer,
)

# trainer.train()
# trainer.push_to_hub(tags=finetune_tags)
# trainer.save_model(str(data_path / finetune_name))

########################################################################################
# Test ORPO Model
########################################################################################

finetune_name = "smol-course-2.20241210T105806"
model_name = f"S1M0N38/{finetune_name}"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt = "Write a haiku about programming"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)  # type: ignore
outputs = model.generate(**inputs, max_new_tokens=100)  # type: ignore

print("After training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

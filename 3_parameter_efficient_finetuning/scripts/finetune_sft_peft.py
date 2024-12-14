########################################################################################
# %% Imports
########################################################################################

import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, AutoPeftModelForCausalLM  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTConfig, SFTTrainer, setup_chat_format

########################################################################################
# %% Load Env Variables
########################################################################################

assert load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
assert DATA_PATH, "Please set the DATA_PATH environment variable"
data_path = Path(DATA_PATH)

HF_HOME = os.getenv("HF_HOME")
assert HF_HOME, "Please set the HF_HOME environment variable"
print(f"Using {HF_HOME} as HF")

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

########################################################################################
# Test Base Model
########################################################################################

prompt = "Write a haiku about programming"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)  # type: ignore
outputs = model.generate(**inputs, max_new_tokens=100)  # type: ignore

print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

finetune_name = f"smol-course-3.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
finetune_tags = ["smol-course", "module_3"]

########################################################################################
# Load Dataset
########################################################################################

subset = "everyday-conversations"
ds = load_dataset(path="HuggingFaceTB/smoltalk", name=subset)

print("Dataset sample:")
print(ds["train"][0])  # type: ignore


token_counts = np.array(
    [
        len(tokenizer.apply_chat_template(sample["messages"]))  # type: ignore
        for sample in ds["train"]  # type: ignore
    ]
)

print(
    "Token counts:\n",
    f"min: {token_counts.min()} | ",
    f"max: {token_counts.max()} | ",
    f"avg: {token_counts.mean().round(2)}",
)

########################################################################################
# Supervised Fine-Tuning (SFT) with TRL using LORA
########################################################################################

lora_config = LoraConfig(
    r=4,  # Rank dimension - typically between 4-32
    lora_alpha=8,  # LoRA scaling factor - typically 2x rank
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

args = SFTConfig(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=1,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    # Attributes for saving and loading
    output_dir=str(data_path / finetune_name),
    hub_model_id=finetune_name,
    run_name=finetune_name,
    # Logging and Eval
    eval_strategy="steps",
    eval_steps=0.1,
    logging_steps=16,
    report_to="wandb",
    max_seq_length=1512,
)


# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=lora_config,
    train_dataset=ds["train"],  # type: ignore
    eval_dataset=ds["test"],  # type: ignore
    processing_class=tokenizer,
)

# trainer.train()
# trainer.push_to_hub(tags=finetune_tags)
# trainer.save_model()


########################################################################################
# Test Base Model + Lora Adapter
########################################################################################

model_name = "S1M0N38/smol-course-3.20241211T171514"
model = AutoPeftModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

prompt = "Write a haiku about programming"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)  # type: ignore
outputs = model.generate(**inputs, max_new_tokens=100)  # type: ignore

print("After training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

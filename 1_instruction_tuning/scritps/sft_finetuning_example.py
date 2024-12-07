import torch
import numpy as np
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format

########################################################################################
# Load Base Model
########################################################################################

assert load_dotenv()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

finetune_name = f"smol-course-1.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
finetune_tags = ["smol-course", "module_1"]


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
# Supervised Fine-Tuning (SFT) with Transformers Reinforcement Learning (TRL)
########################################################################################

sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    hub_model_id=finetune_name,
    max_seq_length=1024,
    report_to="wandb",
    run_name=finetune_name,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],  # type: ignore
    tokenizer=tokenizer,  # type: ignore
    eval_dataset=ds["test"],  # type: ignore
)

trainer.train()
trainer.save_model(f"./{finetune_name}")
trainer.push_to_hub(tags=finetune_tags)


########################################################################################
# Test SFT Model
########################################################################################

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

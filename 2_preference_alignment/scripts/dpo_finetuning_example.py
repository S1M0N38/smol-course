########################################################################################
# %% Imports
########################################################################################

from datetime import datetime

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

########################################################################################
# %% Load Dataset
########################################################################################

assert load_dotenv()

dataset = load_dataset(path="trl-lib/ultrafeedback_binarized")
print(dataset)
print(dataset["train"][0])  # type: ignore


########################################################################################
# %% Load SFT Model
########################################################################################

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model_name = "S1M0N38/smol-course-1.20241207T163021"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = f"smol-course-2.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
finetune_tags = ["smol-course", "module_2"]


########################################################################################
# %% Direct Preference Optimization (DPO) with TRL
########################################################################################

dpo_config = DPOConfig(
    max_steps=2048,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    beta=0.1,
    # Optimize traning for low-end hardware
    # gradient_checkpointing=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=2,
    # torch_empty_cache_steps=8,
    # Learning rate
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    # Tokens
    max_prompt_length=1024,  # input tokens
    max_length=1536,  # input tokens + output tokens
    # Attributes for saving and loading
    output_dir="./smol_dpo_output",
    hub_model_id=finetune_name,
    run_name=finetune_name,
    # Logging and Eval
    logging_steps=8,
    save_strategy="no",
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=64,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset["train"],  # type: ignore
    eval_dataset=dataset["test"].select(range(256)),  # type: ignore
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(f"./{finetune_name}")
trainer.push_to_hub(tags=finetune_tags)

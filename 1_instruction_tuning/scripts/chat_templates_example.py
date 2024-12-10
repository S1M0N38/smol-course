from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from dotenv import load_dotenv
import torch

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


########################################################################################
# Messages: list[dict]
########################################################################################

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {
        "role": "assistant",
        "content": "I'm doing well, thank you! How can I assist you today?",
    },
]


########################################################################################
# Convert list[dict] to string with chatml format
########################################################################################

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print("Conversation with template:\n", input_text)


########################################################################################
# Convert list[dict] to tokens and back to string
########################################################################################

token_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,  # This is the default
    add_generation_prompt=True,  # add a final <|assistant|>
)

print("Tokens:\n", token_ids)
print("Conversation decoded:\n", tokenizer.decode(token_ids))


########################################################################################
# Load dataset
########################################################################################

dataset_name = "HuggingFaceTB/smoltalk"
dataset_subset = "everyday-conversations"
ds = load_dataset(dataset_name, dataset_subset)

print("Dataset sample:\n", ds["train"][0])  # type: ignore


########################################################################################
# Apply chat template to a dataset
########################################################################################


def process_smoltalk(sample):
    sample["template"] = tokenizer.apply_chat_template(
        sample["messages"],
        add_generation_prompt=True,
        tokenize=False,
    )
    return sample


ds = ds.map(process_smoltalk)
print("Dataset sample:\n", ds["train"][0])  # type: ignore

########################################################################################
# Apply chat template to a another dataset
########################################################################################

ds = load_dataset("openai/gsm8k", "main")


def process_gsm8k(sample):
    messages = [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    sample["template"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    return sample


ds = ds.map(process_gsm8k)
print("Dataset sample:\n", ds["train"][0])  # type: ignore

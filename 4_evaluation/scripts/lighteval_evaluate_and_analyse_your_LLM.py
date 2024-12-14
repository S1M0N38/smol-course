########################################################################################
# %% Imports

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import (
    EnvConfig,
    ParallelismManager,
    Pipeline,
    PipelineParameters,
)
from transformers import AutoModelForCausalLM

########################################################################################
# %% Load Env Variables

assert load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
assert DATA_PATH, "Please set the DATA_PATH environment variable"
data_path = Path(DATA_PATH)

HF_HOME = os.getenv("HF_HOME")
assert HF_HOME, "Please set the HF_HOME environment variable"

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set the HF_TOKEN environment variable"


########################################################################################
# %% Evaluation parameters


env_config = EnvConfig()

evaluation_tracker = EvaluationTracker(
    output_dir=str(data_path / "4_evaluation"),
    save_details=False,
    hub_results_org="smol-course-test",
    push_to_hub=True,
    push_to_tensorboard=False,
    public=False,
)

pipeline_params = PipelineParameters(
    env_config=env_config,
    launcher_type=ParallelismManager.ACCELERATE,
    job_id=1,
    override_batch_size=1,
    num_fewshot_seeds=0,
    # max_samples=10,
    use_chat_template=False,
)


########################################################################################
# %% Define domain task (medical domain)

domain_tasks = ",".join(
    [
        "leaderboard|mmlu:anatomy|5|0",
        "leaderboard|mmlu:professional_medicine|5|0",
        "leaderboard|mmlu:high_school_biology|5|0",
        "leaderboard|mmlu:high_school_chemistry|5|0",
    ]
)


########################################################################################
# %% Evaluate Qwen/Qwen2.5-0.5B

qwen_model_name = "Qwen/Qwen2.5-0.5B"
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name)
qwen_pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=qwen_model,
)

qwen_pipeline.evaluate()
qwen_pipeline.save_and_push_results()
qwen_results = qwen_pipeline.get_results()


########################################################################################
# %% Evaluate HuggingFaceTB/SmolLM2-360M-Instruct

smol_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
smol_model = AutoModelForCausalLM.from_pretrained(smol_model_name)
smol_pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=smol_model,
)

smol_pipeline.evaluate()
smol_pipeline.save_and_push_results()
smol_results = smol_pipeline.get_results()


########################################################################################
# %% Load evaluation from HuggingFace and convert to pandas DataFrames


def dataset_name(model_name):
    return (
        f"{evaluation_tracker.hub_results_org}/"
        "details_"
        f"{model_name.replace('/', '__')}"
        f"{'_private' if not evaluation_tracker.public else ''}"
    )


qwen_details = load_dataset(
    dataset_name(qwen_model_name),
    "results",
    split="latest",
)
smol_details = load_dataset(
    dataset_name(smol_model_name),
    "results",
    split="latest",
)

qwen_df = (
    pd.DataFrame.from_records(json.loads(qwen_details["results"][0]))  # type: ignore
    .T["acc"]
    .rename(qwen_model_name)  # type: ignore
)
smol_df = (
    pd.DataFrame.from_records(json.loads(smol_details["results"][0]))  # type: ignore
    .T["acc"]
    .rename(smol_model_name)  # type: ignore
)

df = pd.concat([qwen_df, smol_df], axis=1)
assert isinstance(df, pd.DataFrame)
df.plot(kind="barh")
plt.savefig("plot.png")


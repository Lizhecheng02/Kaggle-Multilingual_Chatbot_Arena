import os
import yaml
from openai import OpenAI

config_path = os.path.join(os.getcwd(), "..", "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]
client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)

train_response = client_openai.files.create(
    file=open("train_data.jsonl", "rb"),
    purpose="fine-tune"
)
print(train_response)
print("The train file id for fine-tuning is:", train_response.id)

val_response = client_openai.files.create(
    file=open("validation_data.jsonl", "rb"),
    purpose="fine-tune"
)
print(val_response)
print("The validation file id for fine-tuning is:", val_response.id)

fine_tune_job = client_openai.fine_tuning.jobs.create(
    training_file=train_response.id,
    validation_file=val_response.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 1}
)
print(fine_tune_job)
print("The job id for fine-tuning is:", fine_tune_job.id)

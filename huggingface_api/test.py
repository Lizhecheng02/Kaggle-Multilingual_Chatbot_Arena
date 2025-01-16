import requests
import yaml
import os


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


config_path = os.path.join(os.getcwd(), "..", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hf_token = config["huggingface"]["token"]

API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-9b-it"
headers = {"Authorization": f"Bearer {hf_token}"}
payload = {
    "inputs": "Today is a great day."
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json()[0]["generated_text"])

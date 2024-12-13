import os
import yaml
import warnings
from openai import OpenAI
warnings.filterwarnings("ignore")

config_path = os.path.join(os.getcwd(), "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]

client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)


def get_response_openai(model, prompt, temperature=0.0, top_p=0.95, max_tokens=1024):
    completion = client_openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    response = completion.choices[0].message.content.strip()
    return response

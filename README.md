### This Repo is for [Kaggle - WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)

#### Python Environment

##### 1. Install Packages

```b
pip install -r requirements.txt
```

##### 2. Create ``config.yaml`` File

```bash
openai:
  api_key: "YOUR_OPENAI_API_KEY"
  organization: "YOUR_OPENAI_ORGANIZATION"
huggingface:
  token: "YOUR_HF_TOKEN"
```

#### Prepare Datasets

##### 1. Set Up Kaggle Env

```bash
export KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
export KAGGLE_KEY="YOUR_KAGGLE_API_KEY"
```

##### 2. Download Datasets

```bash
sudo apt install unzip
kaggle datasets download -d lizhecheng/kaggle-multilingual-chatbot-arena-datasets
unzip kaggle-multilingual-chatbot-arena-datasets.zip
```

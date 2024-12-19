### This Repo is for [Kaggle - WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)

#### Python Environment

##### 1. Install Packages

```b
pip install --upgrade -r requirements.txt
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

- ``Original Competition Dataset``

```bash
sudo apt install unzip
kaggle competitions download -c wsdm-cup-multilingual-chatbot-arena
unzip wsdm-cup-multilingual-chatbot-arena.zip
```

- ``Extra Datasets (also include the original competition dataset)``

```bash
sudo apt install unzip
kaggle datasets download -d lizhecheng/kaggle-multilingual-chatbot-arena-datasets
unzip kaggle-multilingual-chatbot-arena-datasets.zip
```

#### Sequence Classification

##### 1. Gemma Classification

- Customize the parameters in ``gemma2-9b-main.sh``.

```bash
cd gemma_cls
chmod +x ./gemma2-9b-main.sh
python gemma2-9b-main.py
```

#### Fine-tune an OpenAI Model for Pseudo-Labeling

##### 1. Create ``.jsonl`` File.

- Customize the parameters in ``data.sh``.

```bash
cd openai_finetune
chmod +x ./data.sh
python data.py
```

##### 2. Token Number Calculation

```bash
python calculate.py
```

##### 3. Start Fine-tuning

```bash
python finetune.py
```

**Remember your file and job IDs for later use.**

##### 4. Test the Fine-tuned Model

- Set your ``fine_tune_job_id`` in ``test.py``.
- Write the prompt in ``test.py``.

```bash
python test.py
```

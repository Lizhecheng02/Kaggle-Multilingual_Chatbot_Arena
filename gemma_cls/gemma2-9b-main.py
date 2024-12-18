import wandb
import os
import torch
import torch.nn as nn
import argparse
import warnings
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from preprocess import get_train_and_test, get_cls_prompt, form_final_dataset
from sklearn.metrics import accuracy_score
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model
warnings.filterwarnings("ignore")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class CustomTrainer(Trainer):
    def __init__(self, *args, label_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(args):
    ORIGINAL_FILES = args.original_files
    EXTRA_FILES = args.extra_files
    MODEL_NAME = args.model_name
    MAX_LENGTH = args.max_length
    ACCESS_TOKEN = args.access_token
    GRADIENT_CHECKPOINTING = args.gradient_checkpointing
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    WARMUP_RATIO = args.warmup_ratio
    WEIGHT_DECAY = args.weight_decay
    SAVE_TOTAL_LIMIT = args.save_total_limit
    EPOCHS = args.epochs
    STEPS = args.steps
    LR_SCHEDULER = args.lr_scheduler

    wandb.login(key="96a47264bf4a345cddba37487838a3c098362dab")
    run = wandb.init(project=f"lmsys2-{MODEL_NAME.split('/')[-1]}", job_type="training", anonymous="allow")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
    # print(tokenizer.padding_side, tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side, tokenizer.pad_token)

    def tokenize(sample):
        return tokenizer(sample["input"], truncation=True, max_length=MAX_LENGTH)

    train_df, test_df = get_train_and_test(ORIGINAL_FILES, EXTRA_FILES)
    train_df, test_df = get_cls_prompt(train_df, test_df)
    train_ds, test_ds = form_final_dataset(train_df, test_df)
    ds_train = train_ds.map(tokenize).remove_columns(["input"])
    ds_eval = test_ds.map(tokenize).remove_columns(["input"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        token=ACCESS_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.score = nn.Linear(model.config.hidden_size, 2).to(model.device)
    model.config.num_labels = 2
    # print(model.config.pad_token_id)
    model.config.pad_token_id = model.config.eos_token_id
    print(model.config.pad_token_id)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.SEQ_CLS,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "up_proj", "down_proj"
        ]
    )

    lora_model = get_peft_model(model, lora_config)
    print(lora_model)
    print(lora_model.print_trainable_parameters())

    if not GRADIENT_CHECKPOINTING:
        lora_model.gradient_checkpointing_disable()

    class DataCollator:
        def __call__(self, features):
            model_inputs = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"]
                } for feature in features
            ]
            batch = tokenizer.pad(
                model_inputs,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
                pad_to_multiple_of=1
            )
            return batch

    def compute_metrics(p):
        preds, labels = p
        preds = preds.argmax(-1)
        score = accuracy_score(labels, preds)
        return {"accuracy": score}

    training_args = TrainingArguments(
        output_dir=f"{MODEL_NAME.split('/')[-1]}/{wandb.run.name}",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        # optim="paged_adamw_8bit",
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",
        save_steps=STEPS,
        logging_steps=STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_only_model=True,
        lr_scheduler_type=LR_SCHEDULER,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        report_to="wandb"
    )

    trainer = CustomTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLM For Sequence Classification")
    parser.add_argument("--original_files", nargs="+", type=str)
    parser.add_argument("--extra_files", nargs="+", type=str)
    parser.add_argument("--model_name", default="sfairXC/FsfairX-Gemma2-RM-v0.1", type=str)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--gradient_checkpointing", default=False, type=bool)
    parser.add_argument("--access_token", default="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", type=str)
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--accumulation_steps", default=16, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--save_total_limit", default=10, type=int)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    args = parser.parse_args()
    print(args)
    train(args)

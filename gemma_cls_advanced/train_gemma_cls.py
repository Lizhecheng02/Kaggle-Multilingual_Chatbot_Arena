from typing import Any, Dict, List, Optional, Union
from typing import Optional, Union
from transformers import Trainer
from time import gmtime, strftime
from peft import (
    PeftModel,
    get_peft_model,
    LoraConfig,
    TaskType
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
    Gemma2ForSequenceClassification,
    BitsAndBytesConfig
)
from datasets import Dataset
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import argparse
import shutil
import yaml
import wandb
import random
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CustomTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        model_to_save = self.model
        state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
        # Using Hugging Face"s save_pretrained instead of PyTorch"s torch.save
        model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save, safe_serialization=False)

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        o_inputs = inputs.copy()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


def seed_everything(seed=None):
    """
    固定seed
    :param seed: int, 随机种子
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


seed_everything(42)


class TrainInstructionDataSet(Dataset):
    def __init__(self, data, tokenizer, max_source_length, truncation_type="lzc"):
        super(TrainInstructionDataSet, self).__init__()
        self.data = pd.read_parquet(data)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.truncation_type = truncation_type
        self.vis = False
        self.sys_prompt = "You will receive two answers to the user's question, provided by Assistants A and B. Your task is to evaluate and determine which assistant's response is better."
        self.usr_prompt = """[User Question]
{}
[The Start of Assistant A's Answer]
{}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{}
[The End of Assistant B's Answer]"""

    def get_first_and_last(self, text: str, max_length: int, tokenizer) -> str:
        tokenized = tokenizer(
            text=text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False
        )

        input_ids = tokenized["input_ids"]
        offset_mapping = tokenized["offset_mapping"]
        if len(input_ids) <= max_length:
            return text, len(input_ids)

        half_length = (max_length - 5) // 2
        first_half_end = offset_mapping[half_length - 1][1]
        last_half_start = offset_mapping[-half_length][0]

        first_part = text[:first_half_end]
        last_part = text[last_half_start:]

        return first_part + "\n\n<unused1>\n\n" + last_part, max_length

    def count_tokens_gemma(self, text, tokenizer):
        return len(tokenizer.encode(text))

    def truncate_from_tail(self, text, max_tokens, tokenizer):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[-max_tokens:]
        return tokenizer.decode(truncated_tokens)

    def truncate_from_middle(self, text, max_tokens, tokenizer):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens // 2] + tokenizer.encode("...<unused1>...") + tokens[-max_tokens // 2:]
        return tokenizer.decode(truncated_tokens)

    def truncate_by_ratio(self, response_a, response_b, max_total_tokens, tokenizer):
        a_tokens = self.count_tokens_gemma(response_a, tokenizer)
        b_tokens = self.count_tokens_gemma(response_b, tokenizer)
        total_tokens = a_tokens + b_tokens

        if total_tokens <= max_total_tokens:
            return response_a, response_b

        ratio_a = a_tokens / total_tokens if total_tokens != 0 else 0
        a_cut = math.floor(max_total_tokens * ratio_a)
        b_cut = max_total_tokens - a_cut

        truncated_a = self.truncate_from_middle(response_a, a_cut, tokenizer)
        truncated_b = self.truncate_from_middle(response_b, b_cut, tokenizer)
        return truncated_a, truncated_b

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]
        p_t, r_a, r_b = now_data["prompt"], now_data["response_a"], now_data["response_b"]
        winner = now_data["winner"]
        label = 0 if winner == "model_a" else 1

        if self.truncation_type == "znf":
            prefix_length = 90
            content_length = self.max_source_length - prefix_length
            p_t_processed, prompt_length = self.get_first_and_last(p_t, content_length // 5, self.tokenizer)

            response_max_source_length_each = (content_length - prompt_length) // 2

            r_a_t, _ = self.get_first_and_last(r_a, response_max_source_length_each, self.tokenizer)
            r_b_t, _ = self.get_first_and_last(r_b, response_max_source_length_each, self.tokenizer)

            messages = [{"role": "user", "content": self.sys_prompt + "\n" + self.usr_prompt.format(p_t_processed, r_a_t, r_b_t)}]

            prompt_input_ids = self.tokenizer.apply_chat_template(messages, max_length=self.max_source_length, truncation=True)
            input_ids = prompt_input_ids

        elif self.truncation_type == "lzc":
            prompt_tokens = self.count_tokens_gemma(p_t, self.tokenizer)
            response_a_tokens = self.count_tokens_gemma(r_a, self.tokenizer)
            response_b_tokens = self.count_tokens_gemma(r_b, self.tokenizer)

            total_tokens = prompt_tokens + response_a_tokens + response_b_tokens

            total_threshold = self.max_source_length - len(self.tokenizer.encode(self.sys_prompt)) - len(self.tokenizer.encode(self.usr_prompt)) - 20

            if total_tokens <= total_threshold:
                p_t_processed, r_a_t, r_b_t = p_t, r_a, r_b

            if prompt_tokens > total_threshold // 3:
                p_t_processed, r_a_t, r_b_t = p_t, r_a, r_b

            if (response_a_tokens + response_b_tokens) > 3 * prompt_tokens:
                r_a_t, r_b_t = self.truncate_by_ratio(r_a, r_b, total_threshold - prompt_tokens, self.tokenizer)
                p_t_processed = p_t

            if (response_a_tokens + response_b_tokens) < total_threshold:
                p_t_processed = self.truncate_from_middle(p_t, total_threshold - response_a_tokens - response_b_tokens, self.tokenizer)
                r_a_t, r_b_t = r_a, r_b

            else:
                p_t_processed = self.truncate_from_middle(p_t, total_threshold // 6, self.tokenizer)
                r_a_t, r_b_t = self.truncate_by_ratio(r_a, r_b, total_threshold - total_threshold // 6, self.tokenizer)

            messages = [{"role": "user", "content": self.sys_prompt + "\n" + self.usr_prompt.format(p_t_processed, r_a_t, r_b_t)}]

            prompt_input_ids = self.tokenizer.apply_chat_template(messages, max_length=self.max_source_length, truncation=True)
            input_ids = prompt_input_ids

        else:
            raise ValueError("Invalid Truncation Type")

        if self.vis:
            print("=" * 20 + "Train" + "=" * 20)
            print(label)
            print(input_ids)

            print("=" * 20 + "Train Decoding" + "=" * 20)
            input_text = self.tokenizer.decode(input_ids)
            print(input_text)

        return {"input_ids": input_ids, "labels": label}


class ValidInstructionDataSet(Dataset):
    def __init__(self, data, tokenizer, max_source_length, test_mode, truncation_type="lzc"):
        super(ValidInstructionDataSet, self).__init__()
        self.data = pd.read_parquet(data)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        if test_mode:
            self.data = self.data[:10]
        self.truncation_type = truncation_type
        self.vis = False
        self.sys_prompt = "You will receive two answers to the user's question, provided by Assistants A and B. Your task is to evaluate and determine which assistant's response is better."
        self.usr_prompt = """[User Question]
{}
[The Start of Assistant A's Answer]
{}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{}
[The End of Assistant B's Answer]"""

    def get_first_and_last(self, text: str, max_length: int, tokenizer) -> str:
        tokenized = tokenizer(
            text=text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False
        )

        input_ids = tokenized["input_ids"]
        offset_mapping = tokenized["offset_mapping"]
        if len(input_ids) <= max_length:
            return text, len(input_ids)

        half_length = (max_length - 5) // 2
        first_half_end = offset_mapping[half_length - 1][1]
        last_half_start = offset_mapping[-half_length][0]

        first_part = text[:first_half_end]
        last_part = text[last_half_start:]

        return first_part + "\n\n<unused1>\n\n" + last_part, max_length

    def count_tokens_gemma(self, text, tokenizer):
        return len(tokenizer.encode(text))

    def truncate_from_tail(self, text, max_tokens, tokenizer):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[-max_tokens:]
        return tokenizer.decode(truncated_tokens)

    def truncate_from_middle(self, text, max_tokens, tokenizer):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens // 2] + tokenizer.encode("...<unused1>...") + tokens[-max_tokens // 2:]
        return tokenizer.decode(truncated_tokens)

    def truncate_by_ratio(self, response_a, response_b, max_total_tokens, tokenizer):
        a_tokens = self.count_tokens_gemma(response_a, tokenizer)
        b_tokens = self.count_tokens_gemma(response_b, tokenizer)
        total_tokens = a_tokens + b_tokens

        if total_tokens <= max_total_tokens:
            return response_a, response_b

        ratio_a = a_tokens / total_tokens if total_tokens != 0 else 0
        a_cut = math.floor(max_total_tokens * ratio_a)
        b_cut = max_total_tokens - a_cut

        truncated_a = self.truncate_from_middle(response_a, a_cut, tokenizer)
        truncated_b = self.truncate_from_middle(response_b, b_cut, tokenizer)
        return truncated_a, truncated_b

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]
        p_t, r_a, r_b = now_data["prompt"], now_data["response_a"], now_data["response_b"]
        winner = now_data["winner"]
        label = 0 if winner == "model_a" else 1

        if self.truncation_type == "znf":
            prefix_length = 90
            content_length = self.max_source_length - prefix_length
            p_t_processed, prompt_length = self.get_first_and_last(p_t, content_length // 5, self.tokenizer)

            response_max_source_length_each = (content_length - prompt_length) // 2

            r_a_t, _ = self.get_first_and_last(r_a, response_max_source_length_each, self.tokenizer)
            r_b_t, _ = self.get_first_and_last(r_b, response_max_source_length_each, self.tokenizer)

            messages = [{"role": "user", "content": self.sys_prompt + "\n" + self.usr_prompt.format(p_t_processed, r_a_t, r_b_t)}]

            prompt_input_ids = self.tokenizer.apply_chat_template(messages, max_length=self.max_source_length, truncation=True)
            input_ids = prompt_input_ids

        elif self.truncation_type == "lzc":
            prompt_tokens = self.count_tokens_gemma(p_t, self.tokenizer)
            response_a_tokens = self.count_tokens_gemma(r_a, self.tokenizer)
            response_b_tokens = self.count_tokens_gemma(r_b, self.tokenizer)

            total_tokens = prompt_tokens + response_a_tokens + response_b_tokens

            total_threshold = self.max_source_length - len(self.tokenizer.encode(self.sys_prompt)) - len(self.tokenizer.encode(self.usr_prompt)) - 20

            if total_tokens <= total_threshold:
                p_t_processed, r_a_t, r_b_t = p_t, r_a, r_b

            if prompt_tokens > total_threshold // 3:
                p_t_processed, r_a_t, r_b_t = p_t, r_a, r_b

            if (response_a_tokens + response_b_tokens) > 3 * prompt_tokens:
                r_a_t, r_b_t = self.truncate_by_ratio(r_a, r_b, total_threshold - prompt_tokens, self.tokenizer)
                p_t_processed = p_t

            if (response_a_tokens + response_b_tokens) < total_threshold:
                p_t_processed = self.truncate_from_middle(p_t, total_threshold - response_a_tokens - response_b_tokens, self.tokenizer)
                r_a_t, r_b_t = r_a, r_b

            else:
                p_t_processed = self.truncate_from_middle(p_t, total_threshold // 6, self.tokenizer)
                r_a_t, r_b_t = self.truncate_by_ratio(r_a, r_b, total_threshold - total_threshold // 6, self.tokenizer)

            messages = [{"role": "user", "content": self.sys_prompt + "\n" + self.usr_prompt.format(p_t_processed, r_a_t, r_b_t)}]

            prompt_input_ids = self.tokenizer.apply_chat_template(messages, max_length=self.max_source_length, truncation=True)
            input_ids = prompt_input_ids

        else:
            raise ValueError("Invalid Truncation Type")

        if self.vis:
            print("=" * 20 + "Valid" + "=" * 20)
            print(label)
            print(input_ids)

            print("=" * 20 + "Valid Decoding" + "=" * 20)
            input_text = self.tokenizer.decode(input_ids)
            print(input_text)

        return {"input_ids": input_ids, "labels": label}


@dataclass
class DataCollatorForClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]

        # Flatten the features (no need to handle multiple choices)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if batch["input_ids"].dim() == 3:
            batch["input_ids"] = batch["input_ids"].squeeze(1)
        if batch["input_ids"].dim() == 1:
            batch["input_ids"] = batch["input_ids"].unsqueeze(0)

        if "token_type_ids" in batch:
            if batch["token_type_ids"].dim() == 3:
                batch["token_type_ids"] = batch["token_type_ids"].squeeze(1)
            if batch["token_type_ids"].dim() == 1:
                batch["token_type_ids"] = batch["token_type_ids"].unsqueeze(0)

        if "attention_mask" in batch:
            if batch["attention_mask"].dim() == 3:
                batch["attention_mask"] = batch["attention_mask"].squeeze(1)
            if batch["attention_mask"].dim() == 1:
                batch["attention_mask"] = batch["attention_mask"].unsqueeze(0)

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(p):
    logits = p.predictions
    predictions = np.argmax(logits, axis=-1)
    labels = p.label_ids
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train(args):
    s = strftime("%a_%d_%b_%H_%M", gmtime())

    wandb.init(project="WSDM CUP", config=args, name=f"{args.run_name}")
    wandb.save("train_gemma_cls.py")
    wandb.save("train_gemma_cls.yaml")

    wandb.save(args.config)
    MODEL = args.MODEL

    args.output_dir = f"{args.output_dir}/{wandb.run.name}"
    os.makedirs(args.output_dir, exist_ok=True)

    shutil.copy2("train_gemma_cls.py", args.output_dir)
    shutil.copy2(args.config, args.output_dir)

    # process dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, truncation_side="left")
    tokenizer.padding_side = "right"

    tokenized_dataset = TrainInstructionDataSet(args.train_data, tokenizer, args.MAX_INPUT, truncation_type=args.truncation_type)
    tokenized_dataset_valid = ValidInstructionDataSet(args.valid_data, tokenizer, args.MAX_INPUT, args.test_mode)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Gemma2ForSequenceClassification.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    print(f"AutoModelForSequenceClassification is {model}")
    model.score = nn.Linear(model.config.hidden_size, 2, bias=False).to(model.device, torch.bfloat16)
    model.config.num_labels = 2

    print("disable softcapping")
    model.config.attn_logit_softcapping = None
    model.config.use_cache = False

    checkpoint = None
    if len(args.resume_from_checkpoint) != 0:
        checkpoint = args.resume_from_checkpoint
        print(f"Using checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        new_score = torch.nn.Linear(in_features=3584, out_features=2, bias=False).to(model.device, torch.bfloat16)
        model.base_model.model.score.modules_to_save.default = new_score
        for param in model.base_model.model.score.modules_to_save.default.parameters():
            param.requires_grad = True

    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )

        model = get_peft_model(model, peft_config)
    print(f"model is {model}")
    print(model.print_trainable_parameters())

    data_collator = DataCollatorForClassification(
        tokenizer,
        max_length=args.MAX_INPUT,
        pad_to_multiple_of=None,
        padding=True,
    )

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        report_to="wandb",
        overwrite_output_dir=True,
        greater_is_better=True,
        bf16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps if not args.test_mode else 1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps if not args.test_mode else 1,
        save_strategy="steps",
        save_steps=args.save_steps if not args.test_mode else 1,
        save_only_model=True,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        save_total_limit=50,
        label_smoothing_factor=args.label_smoothing_factor
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    num_warmup_steps = int(args.warmup_ratio * training_args.num_train_epochs * int(len(tokenized_dataset) * 1.0 / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count())))
    num_training_steps = training_args.num_train_epochs * (len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()
    wandb.finish()


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument("--config", default="./train_gemma_cls.yaml", type=str, help="Path to the config file", required=False)
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()
    train(args)

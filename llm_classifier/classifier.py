import os
import logging
import torch
import evaluate
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from llama_finetuning.accelerator import accelerator
from llm_classifier.prepare_dataset import train_dataset, eval_dataset, train_text, train_label

from llama_finetuning.llm import LargeLanguageModel, rename_finetuned_model_path

from conf import configuration


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class LargeLanguageModelClassifier(LargeLanguageModel):

    def __init__(self):
        super().__init__()

    def init_conf(self):
        self.config = LoraConfig(
            r=configuration.lora_r,
            lora_alpha=configuration.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="SEQ_CLS",
        )
        if configuration.load_in_kbit == 4:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif configuration.load_in_kbit == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=0.0
            )
        else:
            raise Exception('The value for load_in_kbit in configuration file is incorrect')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=2,
            quantization_config=self.bnb_config,
            use_cache=False
        )
        self.model = prepare_model_for_kbit_training(self.model)
        logging.info(self.model)
        self.model = get_peft_model(self.model, self.config)
        self.model = accelerator.prepare_model(self.model)
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def init_trainer(self):
        self.tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt2)
        self.tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt2)
        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            compute_metrics=compute_metrics,
            args=transformers.TrainingArguments(
                output_dir=os.path.join(configuration.learned_models, configuration.base_model_id),
                warmup_steps=1,
                per_device_train_batch_size=configuration.batch_size,
                per_device_eval_batch_size=configuration.batch_size,
                gradient_accumulation_steps=1,
                max_steps=configuration.max_steps,
                learning_rate=2.5e-5,
                fp16=True,
                optim="paged_adamw_8bit",
                logging_dir=configuration.logging_folder,
                save_strategy="steps",
                save_steps=configuration.save_steps,
                evaluation_strategy="steps",
                eval_steps=configuration.eval_steps,
                do_eval=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': True}
            ),
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer),
        )

    def generate_and_tokenize_prompt(self, prompt):
        return self.tokenizer(train_text(prompt))

    def generate_and_tokenize_prompt2(self, prompt):
        result = self.tokenizer(
            train_text(prompt),
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
        )
        result["labels"] = train_label(prompt)
        return result


if __name__ == '__main__':
    initialization()
    obj = LargeLanguageModelClassifier()
    obj.init_conf()
    obj.init_model()
    # obj.determine_tokenizer_max_length()
    obj.init_trainer()
    obj.finetuning()
    rename_finetuned_model_path()

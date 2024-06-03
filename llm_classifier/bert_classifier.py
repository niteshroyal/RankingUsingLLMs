import os
import logging
import evaluate
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llama_finetuning.accelerator import accelerator
from llm_classifier.prepare_dataset import train_dataset, eval_dataset, train_text, train_label

from llama_finetuning.llm import rename_finetuned_model_path

from utils.utils import plot_data_lengths

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


class BERTLanguageModel:

    def __init__(self):
        self.config = None
        self.tokenized_val_dataset = None
        self.tokenized_train_dataset = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model_id = configuration.base_model_id

    def init_conf(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="right",
            add_special_tokens=True,
        )

    def init_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=2,
        )
        # self.model = accelerator.prepare_model(self.model)

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
                learning_rate=2e-5,
                fp16=True,
                logging_dir=configuration.logging_folder,
                save_strategy="steps",
                save_steps=configuration.save_steps,
                evaluation_strategy="steps",
                eval_steps=configuration.eval_steps,
                do_eval=True,
            ),
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer),
        )

    def finetuning(self):
        self.trainer.train()

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

    def determine_tokenizer_max_length(self):
        self.tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt)
        self.tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt)
        plot_data_lengths(self.tokenized_train_dataset, self.tokenized_val_dataset)


if __name__ == '__main__':
    initialization()
    obj = BERTLanguageModel()
    obj.init_conf()
    obj.init_model()
    # obj.determine_tokenizer_max_length()
    obj.init_trainer()
    obj.finetuning()
    rename_finetuned_model_path()

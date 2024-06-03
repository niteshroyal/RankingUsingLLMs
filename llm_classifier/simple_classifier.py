import os
import logging
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

from llm_classifier.prepare_dataset import train_dataset, eval_dataset, train_text, train_label
from llama_finetuning.llm import rename_finetuned_model_path
from conf import configuration


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class LargeLanguageModelSimpleClassifier:

    def __init__(self):
        self.tokenized_val_dataset = None
        self.tokenized_train_dataset = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model_id = configuration.base_model_id

    def init_model(self):
        peft_model_id = "ybelkada/opt-350m-lora"
        model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)


        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model_id,
            load_in_4bit=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_trainer(self):
        self.tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt2)
        self.tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt2)
        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=os.path.join(configuration.learned_models, configuration.base_model_id),
                warmup_steps=1,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
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


if __name__ == '__main__':
    initialization()
    obj = LargeLanguageModelSimpleClassifier()
    obj.init_model()
    obj.init_trainer()
    obj.finetuning()
    rename_finetuned_model_path()

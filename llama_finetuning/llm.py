import os
import shutil
import logging
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from llama_finetuning.accelerator import accelerator
from llama_finetuning.load_finetuning_dataset import train_dataset, eval_dataset, formatting_func_for_train

from conf import configuration
from utils.utils import plot_data_lengths


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_generator_llm():
    base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset))[0]
    generator_llm = f"checkpoint{configuration.max_steps}-run-{configuration.run_id}-{base_file_name}"

    # generator_llm = (f"checkpoint{configuration.max_steps}-"
    #                  f"top{configuration.topn}{configuration.entity_type}-"
    #                  f"ntraining{configuration.num_training_datapoints}")

    generator_llm = os.path.join(os.path.join(configuration.learned_models, configuration.base_model_id), generator_llm)
    return generator_llm


def get_old_generator_llm_location():
    generator_llm = f"checkpoint-{configuration.max_steps}"
    generator_llm = os.path.join(os.path.join(configuration.learned_models, configuration.base_model_id), generator_llm)
    return generator_llm


def rename_finetuned_model_path():
    old_path = get_old_generator_llm_location()
    new_path = get_generator_llm()
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"The folder '{old_path}' does not exist.")
    if os.path.exists(new_path) and os.listdir(new_path):
        shutil.rmtree(new_path)
    os.rename(old_path, new_path)
    logging.info(f"Finetuning complete. The finetuned model can be found here: {new_path}")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


class LargeLanguageModel:

    def __init__(self):
        self.config = None
        self.bnb_config = None
        self.tokenized_val_dataset = None
        self.tokenized_train_dataset = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model_id = configuration.base_model_id

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
            task_type="CAUSAL_LM",
        )
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id, quantization_config=self.bnb_config)
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        logging.info(self.model)
        self.model = get_peft_model(self.model, self.config)
        self.model = accelerator.prepare_model(self.model)

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
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

    def finetuning(self):
        self.trainer.train()

    def generate_and_tokenize_prompt(self, prompt):
        return self.tokenizer(formatting_func_for_train(prompt))

    def generate_and_tokenize_prompt2(self, prompt):
        result = self.tokenizer(
            formatting_func_for_train(prompt),
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def determine_tokenizer_max_length(self):
        self.tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt)
        self.tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt)
        plot_data_lengths(self.tokenized_train_dataset, self.tokenized_val_dataset)


if __name__ == '__main__':
    initialization()
    obj = LargeLanguageModel()
    obj.init_conf()
    obj.init_model()
    # obj.determine_tokenizer_max_length()
    obj.init_trainer()
    obj.finetuning()
    rename_finetuned_model_path()

import os
import logging
from transformers import AutoTokenizer

from llama_finetuning.load_finetuning_dataset import train_dataset, eval_dataset, formatting_func_for_train

from conf import configuration
from utils.utils import plot_data_lengths


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class AnalyzeTokenizer:

    def __init__(self):
        self.tokenized_val_dataset = None
        self.tokenized_train_dataset = None
        self.tokenizer = None
        self.base_model_id = configuration.base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_and_tokenize_prompt(self, prompt):
        return self.tokenizer(formatting_func_for_train(prompt))

    def determine_tokenizer_max_length(self):
        self.tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt)
        self.tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt)
        plot_data_lengths(self.tokenized_train_dataset, self.tokenized_val_dataset)


if __name__ == '__main__':
    initialization()
    obj = AnalyzeTokenizer()
    obj.determine_tokenizer_max_length()

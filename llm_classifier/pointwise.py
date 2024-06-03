import os
import logging
import torch
import random
import evaluate
import shutil
import transformers
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from llama_finetuning.accelerator import accelerator
from llm_classifier.prepare_dataset import (train_dataset, eval_dataset, train_text, train_rank, train_group,
                                            pointwise_eval_label, pointwise_eval_text)
from llm_classifier.loss_functions import compute_minibatch_loss
from conf import configuration


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


metric = evaluate.load("accuracy")


def get_old_generator_llm_location():
    generator_llm = f"pointwise-checkpoint-{configuration.save_steps}"
    generator_llm = os.path.join(os.path.join(configuration.learned_models, configuration.base_model_id), generator_llm)
    return generator_llm


def process_a_eval_batch(batch):
    input_ids1_list = []
    attention_mask1_list = []
    input_ids2_list = []
    attention_mask2_list = []
    labels_list = []
    for item in batch:
        input_ids1_list.append(item['input_ids1'])
        attention_mask1_list.append(item['attention_mask1'])
        input_ids2_list.append(item['input_ids2'])
        attention_mask2_list.append(item['attention_mask2'])
        labels_list.append(item['labels'])
    input_ids1 = torch.stack(input_ids1_list, dim=0)
    attention_mask1 = torch.stack(attention_mask1_list, dim=0)
    input_ids2 = torch.stack(input_ids2_list, dim=0)
    attention_mask2 = torch.stack(attention_mask2_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {
        'input_ids1': input_ids1,
        'attention_mask1': attention_mask1,
        'input_ids2': input_ids2,
        'attention_mask2': attention_mask2,
        'labels': labels
    }


class FixedSizeLossQueue:
    def __init__(self, size=100):
        self.size = size
        self.queue = deque([0] * size, maxlen=size)
        self.elements_added = 0

    def add(self, item):
        self.elements_added += 1
        self.queue.append(item)

    def get_queue(self):
        return list(self.queue)

    def average(self):
        if self.elements_added == 0:
            return 0
        actual_elements = min(self.elements_added, self.size)
        return sum(self.queue) / actual_elements


class EvalDatasetLoader:
    def __init__(self, datapoints, tokenizer):
        self.tokenizer = tokenizer
        self.datapoints = datapoints
        self.batches = []
        self.token_sizes = []

    def len(self):
        return len(self.datapoints)

    def get_tokenized_data(self, datapoint):
        [prompt1, prompt2] = pointwise_eval_text(datapoint)
        label = pointwise_eval_label(datapoint)
        result1 = self.tokenizer(
            prompt1,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result2 = self.tokenizer(
            prompt2,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.token_sizes.append(len(self.tokenizer(prompt1)['input_ids']))
        self.token_sizes.append(len(self.tokenizer(prompt2)['input_ids']))
        return {
            'input_ids1': result1['input_ids'].flatten(),
            'attention_mask1': result1['attention_mask'].flatten(),
            'input_ids2': result2['input_ids'].flatten(),
            'attention_mask2': result2['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def process(self):
        processed_datapoints = []
        for datapoint in self.datapoints:
            tokenized = self.get_tokenized_data(datapoint)
            processed_datapoints.append(tokenized)
        assert configuration.batch_size % 2 == 0, "batch_size is not even"
        validation_batch_size = int(configuration.batch_size / 2)
        batch = []
        for datapoint in processed_datapoints:
            if len(batch) == validation_batch_size:
                self.batches.append(batch)
                batch = [datapoint]
            else:
                batch.append(datapoint)
        if batch:
            self.batches.append(batch)


class TrainDatasetLoader:
    def __init__(self, datapoints, tokenizer):
        self.tokenizer = tokenizer
        self.datapoints = datapoints
        self.groups = None
        self.processed_datapoints = dict()
        self.rng = random.Random(42)
        self.token_sizes = []

    def len(self):
        return len(self.datapoints)

    def get_tokenized_data(self, datapoint):
        prompt = train_text(datapoint)
        rank = train_rank(datapoint)
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.token_sizes.append(len(self.tokenizer(prompt)['input_ids']))
        return {
            'input_ids': result['input_ids'].flatten(),
            'attention_mask': result['attention_mask'].flatten(),
            'labels': torch.tensor(rank, dtype=torch.float)
        }

    def process(self):
        groups = set()
        for datapoint in self.datapoints:
            groups.add(train_group(datapoint))
        self.groups = list(groups)
        for group in self.groups:
            self.processed_datapoints[group] = []
        for datapoint in self.datapoints:
            group = train_group(datapoint)
            tokenized = self.get_tokenized_data(datapoint)
            self.processed_datapoints[group].append(tokenized)

    def get_a_minibatch(self):
        a_random_group = self.rng.choice(self.groups)
        my_list = self.processed_datapoints[a_random_group]
        num_elements_to_sample = min(len(my_list), configuration.batch_size)
        if len(my_list) <= configuration.batch_size:
            shuffled_elements = my_list[:]
            random.shuffle(shuffled_elements)
        else:
            sampled_elements = random.sample(my_list, num_elements_to_sample)
            shuffled_elements = sampled_elements
            random.shuffle(shuffled_elements)
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in shuffled_elements:
            input_ids_list.append(item['input_ids'])
            attention_mask_list.append(item['attention_mask'])
            labels_list.append(item['labels'])
        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class LargeLanguageModelPointWise:

    def __init__(self):
        self.config = None
        self.bnb_config = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model_id = configuration.base_model_id
        self.device = None
        self.training_data_loader = None
        self.eval_data_loader = None
        self.loss_name = 'bce'
        self.training_loss_queue = FixedSizeLossQueue(100)

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
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
        else:
            logging.info("No GPU detected\n")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=1,
            quantization_config=self.bnb_config,
            use_cache=False
        )
        self.model = prepare_model_for_kbit_training(self.model)
        logging.info(self.model)
        self.model = get_peft_model(self.model, self.config)
        self.model = accelerator.prepare_model(self.model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.to(self.device)

    def eval(self):
        counter = 0
        self.model.eval()
        references = torch.randn(0).to(self.device)
        predictions = torch.randn(0).to(self.device)
        with torch.no_grad():
            for batch in self.eval_data_loader.batches:
                batch = process_a_eval_batch(batch)
                input_ids1 = batch['input_ids1']
                attention_mask1 = batch['attention_mask1']
                input_ids2 = batch['input_ids2']
                attention_mask2 = batch['attention_mask2']
                input_ids = torch.cat((input_ids1, input_ids2), dim=0).to(self.device)
                attention_mask = torch.cat((attention_mask1, attention_mask2), dim=0).to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                scores1 = logits[:input_ids1.size(0)]
                scores2 = logits[input_ids1.size(0):]
                prediction = (scores1 > scores2).float().squeeze()
                references = torch.cat((references, labels), dim=0)
                predictions = torch.cat((predictions, prediction), dim=0)
                counter += input_ids1.size(0)
                logging.info(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')
                print(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')
        result = metric.compute(predictions=predictions, references=references)
        logging.info(f"Average pairwise eval_accuracy is {result['accuracy']}")
        print(f"Average pairwise eval_accuracy is {result['accuracy']}")
        self.model.train()

    def init_trainer(self):
        self.training_data_loader = TrainDatasetLoader(train_dataset, self.tokenizer)
        self.training_data_loader.process()
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()
        self.trainer = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                output_dir=get_old_generator_llm_location()
            )
        )

    def finetuning(self):
        print(f"There are {self.training_data_loader.len()} datapoints in the training dataset")
        logging.info(f"There are {self.training_data_loader.len()} datapoints in the training dataset")
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=2.5e-5)
        self.model.train()
        max_steps_counter = 0
        number_of_training_datapoints_till_now = 0
        while max_steps_counter < configuration.max_steps:
            batch = self.training_data_loader.get_a_minibatch()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            ranks = batch['labels'].to(self.device)
            number_of_training_datapoints_till_now += input_ids.size(0)
            max_steps_counter += 1
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = compute_minibatch_loss(logits, ranks, self.loss_name)
            loss.backward()
            optimizer.step()
            self.training_loss_queue.add(loss.item())
            print(f"Till now {number_of_training_datapoints_till_now} datapoints sampled for training, "
                  f"Loss: {loss.item()}, "
                  f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                  f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            logging.info(f"Till now {number_of_training_datapoints_till_now} datapoints sampled for training, "
                         f"Loss: {loss.item()}, "
                         f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                         f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            if max_steps_counter % configuration.eval_steps == 0:
                self.eval()
            if max_steps_counter == configuration.save_steps:
                self.trainer.save_model()

    def determine_tokenizer_max_length(self):
        lengths = self.training_data_loader.token_sizes
        lengths += self.eval_data_loader.token_sizes

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Number of tokens')
        plt.ylabel('Frequency')
        plt.title('Distribution of number of tokens')
        plt.show()

    def generate_and_tokenize_prompt(self, prompt):
        return self.tokenizer(train_text(prompt))

    def get_generator_llm(self):
        if self.loss_name == 'bce':
            mark = ''
        elif self.loss_name == 'bce_same_rank_handling':
            mark = '-same-ranks-differently'
        elif self.loss_name == 'hinge':
            mark = '-hinge-loss'
        else:
            raise Exception('Loss name incorrect')

        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset))[0]
        generator_llm = f"pointwise{mark}-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, configuration.base_model_id),
                                     generator_llm)
        return generator_llm

    def rename_finetuned_model_path(self):
        old_path = get_old_generator_llm_location()
        new_path = self.get_generator_llm()
        if not os.path.exists(old_path):
            raise FileNotFoundError(f"The folder '{old_path}' does not exist.")
        if os.path.exists(new_path) and os.listdir(new_path):
            shutil.rmtree(new_path)
        os.rename(old_path, new_path)
        logging.info(f"Finetuning complete. The finetuned model can be found here: {new_path}")


if __name__ == '__main__':
    initialization()
    obj = LargeLanguageModelPointWise()
    obj.init_conf()
    obj.init_model()
    obj.init_trainer()
    # obj.determine_tokenizer_max_length()
    obj.finetuning()
    obj.rename_finetuned_model_path()

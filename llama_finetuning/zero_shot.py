import os
import torch
import random
import logging
from llama_finetuning.llm import LargeLanguageModel

from transformers import AutoModelForCausalLM

from conf import configuration
from llm_classifier.eval_classifier import evaluate_on_validation_dataset


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class ZeroShot(LargeLanguageModel):

    # def init_model(self):
    #     self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id).to('cuda')
    #     logging.info(self.model)

    def __init__(self):
        super().__init__()
        self.init_conf()
        self.init_model()

    def evaluate(self, eval_prompts, tokenizer_max_length=configuration.tokenizer_max_length):
        new_eval_prompts = []
        for prompt in eval_prompts:
            new_eval_prompts.append('Answer the following with Yes or No only. In the worst case, if you do not know '
                                    'the answer then choose randomly between Yes and No.\n'
                                    'This question is about two rivers: Is Nile longer than Indus?\nYes\n'
                                    'This question is about two countries: Is Japan more populated than India?\nNo\n' +
                                    'This question is about two countries: Is China larger than India?\nYes\n' +
                                    prompt + '\n')
        inputs = self.tokenizer(
            new_eval_prompts,
            truncation=True,
            max_length=tokenizer_max_length + 40,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.model.eval()
        with torch.no_grad():
            # outputs = self.model.generate(**inputs, max_new_tokens=2,
            #                               return_dict_in_generate=True, output_scores=True)
            # score_for_yes = outputs.scores[1][:, 9904]
            # score_for_no = outputs.scores[1][:, 3084]
            # prediction = (score_for_yes > score_for_no).int()

            g_str = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=2)

            prediction = []
            for i in range(0, len(g_str)):
                st = self.tokenizer.decode(g_str[i], skip_special_tokens=True)
                logging.info(st)
                pred = process_string(st)
                logging.info(f'Answer part in process string = {pred}')
                prediction.append(pred)
            prediction = torch.tensor(prediction)
        return prediction


# def process_string(input_string):
#     # Split the string at "\n"
#     parts = input_string.split("\n")
#
#     if len(parts) > 1:
#         answer_part = parts[8].lower().strip()
#         if "yes" in answer_part:
#             return 1
#         elif "no" in answer_part:
#             return 0
#     logging.info(f'Answer part in process string = {answer_part}')
#     # If neither "Yes" nor "No" is present, randomly choose between 1 and 0
#     return random.randint(0, 1)


def process_string(text):
    lines = text.strip().split('\n')
    if len(lines) > 8:
        target_line = lines[8].strip()
        first_word = target_line.split()[0].lower()
        if "yes" in first_word:
            return 1
        elif "no" in first_word:
            return 0
        else:
            pass
    return random.randint(0, 1)


if __name__ == '__main__':
    initialization()
    # questions = ['This question is about a river: Is Nile among the longest rivers?',
    #              'This question is about a river: Is Amazon among the longest rivers?',
    #              'This question is about a river: Is Volga among the longest rivers?',
    #              'This question is about a river: Is Ganges among the longest rivers?',
    #              'This question is about a river: Is Tigris among the longest rivers?',
    #              'This question is about a river: Is Meuse among the longest rivers?',
    #              'This question is about a river: Is Elbe among the longest rivers?',
    #              'This question is about a river: Is Rubicon among the longest rivers?']
    # obj = ZeroShot()
    # answers = obj.evaluate(questions)
    # print(answers)

    obj = ZeroShot()

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_wikidata_main_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_wikidata_subtle_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_food_taste_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
              'bitterest food items', 'umami food items', 'fattiest food items']
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_rocks_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
              'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
              'densest rocks']
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_movies_100_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_validation_new_books_100_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"sizes_validation.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"heights_validation.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"masses_validation.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

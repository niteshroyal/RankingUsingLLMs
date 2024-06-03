import os
import json
import logging
import random

from conf import configuration
from utils.openai_api import OpenAI


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + f'-all-at-once.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


# def get_prompt():
#     prompt = (
#         f"Answer the following with Yes or No only. If unsure, choose randomly between Yes and No."
#     )
#     return prompt

def get_prompt():
    prompt = (
        f"Answer the following with Yes or No only. In the worst case, if you don't know the answer then choose "
        f"randomly between Yes and No."
    )
    return prompt


def get_finalprompt(question):
    prompt = get_prompt()
    finalprompt = (
        f"{prompt}\n\n"
        f"{question}"
    )
    return finalprompt


def extract_answer(text):
    text = text.strip()
    text = text.replace('.', '')
    return text


def get_answer(response):
    parts = response.split("### Answer:")
    if len(parts) < 2:
        return "No answer section found."
    response_section = parts[1].split("### End")[0]
    return response_section.strip()


class EvaluateGPT4(OpenAI):

    def __init__(self, model=None):
        super().__init__(model)
        random.seed(42)

    def evaluate_on_validation_dataset(self, considered_group=None):
        hits = []
        counter = 0
        no_answer = 0
        with open(configuration.validation_dataset, 'r') as f:
            for line in f:
                datapoint = json.loads(line)
                if considered_group is None:
                    pass
                else:
                    if datapoint['group'] != considered_group:
                        continue
                prompt = get_finalprompt(datapoint["question"])
                response = self.get_gpt_response(prompt)
                # print(response + '\t' + datapoint['answer'])
                answer = extract_answer(response)
                logging.info(f"Question: {datapoint['question']}, Answer: {datapoint['answer']}, "
                             f"Prediction: {answer}, Response: {response}")
                if answer not in ['Yes', 'No']:
                    answer = random.choice(['Yes', 'No'])
                    no_answer += 1
                else:
                    pass
                if datapoint['answer'] == answer:
                    hits.append(1)
                else:
                    hits.append(0)
                counter += 1
                if counter % 50 == 0:
                    logging.info(f"Number of validation datapoints processed = {counter}, "
                                 f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
                    print(f"Number of validation datapoints processed = {counter}, "
                          f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
        accuracy = (sum(hits) * 100) / len(hits)
        logging.info(f'Final pairwise accuracy for {considered_group} is {accuracy}%, '
                     f'Total number of datapoints processed is {counter}, No answer = {no_answer}/{counter}, '
                     f'Validation file = {configuration.validation_dataset}')
        print(f'Final pairwise accuracy for {considered_group} is {accuracy}%, '
              f'Total number of datapoints processed is {counter}, No answer = {no_answer}/{counter}, '
              f'Validation file = {configuration.validation_dataset}')


if __name__ == '__main__':
    initialization()

    models = ["gpt-4-0613", "gpt-3.5-turbo-0613"]
    # models = ["gpt-4-0613"]

    for model in models:
        obj = EvaluateGPT4(model)

        logging.info(f'OpenAI model is set to {model}')

        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_wikidata_main_500_datapoints.jsonl")
        # obj.evaluate_on_validation_dataset()

        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_wikidata_subtle_500_datapoints.jsonl")
        # obj.evaluate_on_validation_dataset()

        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_books_100_datapoints.jsonl")
        # obj.evaluate_on_validation_dataset()
        #
        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_movies_100_datapoints.jsonl")
        # obj.evaluate_on_validation_dataset()
        #
        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_food_taste_500_datapoints.jsonl")
        #
        # groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
        #           'bitterest food items', 'umami food items', 'fattiest food items']
        #
        # for group in groups:
        #     obj.evaluate_on_validation_dataset(group)
        #
        # configuration.validation_dataset = (
        #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        #     f"llm_validation_new_rocks_500_datapoints.jsonl")
        #
        # groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
        #           'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
        #           'densest rocks']
        #
        # for group in groups:
        #     obj.evaluate_on_validation_dataset(group)

        configuration.validation_dataset = (
            f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
            f"sizes_validation.jsonl")
        obj.evaluate_on_validation_dataset()

        configuration.validation_dataset = (
            f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
            f"heights_validation.jsonl")
        obj.evaluate_on_validation_dataset()

        configuration.validation_dataset = (
            f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
            f"masses_validation.jsonl")
        obj.evaluate_on_validation_dataset()

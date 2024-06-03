import os
import sys
import json
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
from llama_finetuning.llm import get_generator_llm
from llm_classifier.classifier import LargeLanguageModelClassifier

from conf import configuration
from llm_classifier.prepare_dataset import train_text, train_label


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] +
                            '_' + sys.argv[1] + '_.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class EvalLargeLanguageModel(LargeLanguageModelClassifier):

    def __init__(self):
        super().__init__()
        self.fine_tuned_model_location = None
        self.ft_model = None
        self.init_conf()

    def init_ft_model(self):
        self.fine_tuned_model_location = get_generator_llm()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=2,
            quantization_config=self.bnb_config
        )
        self.ft_model = PeftModel.from_pretrained(base_model, self.fine_tuned_model_location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id

    def evaluate(self, eval_prompts, max_new_tokens=configuration.tokenizer_max_length):
        model_input = self.tokenizer(
            eval_prompts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            eval_logits = eval_outputs.logits
            eval_predictions = torch.argmax(eval_logits, dim=1)
            return eval_predictions


# def evaluate_on_validation_dataset(tester):
#     hits = []
#     counter = 0
#     with open(configuration.validation_dataset, 'r') as f:
#         for line in f:
#             datapoint = json.loads(line)
#             text = train_text(datapoint)
#             answer = train_label(datapoint)
#             prediction = tester.evaluate(text)
#             if (prediction == 0).item():
#                 p_answer = 0
#             else:
#                 p_answer = 1
#             if answer == p_answer:
#                 hits.append(1)
#             else:
#                 hits.append(0)
#             counter += 1
#             logging.info(f"Answer: {answer}, Predicted answer: {p_answer}")
#             if counter % 10 == 0:
#                 text = (f"Number of validation datapoints processed = {counter}, "
#                         f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
#                 logging.info(text)
#                 print(text)
#     accuracy = (sum(hits) * 100) / len(hits)
#     text = f"Final accuracy is {accuracy}%"
#     logging.info(text)
#     print(text)

new_datapoints = []


def write_results():
    global new_datapoints
    file_name = configuration.validation_dataset
    path, extension = file_name.rsplit('.', 1)
    file_name = f"{path}_results.{extension}"
    with open(file_name, 'w', encoding='utf-8') as file_handler:
        for item in new_datapoints:
            json.dump(item, file_handler)
            file_handler.write('\n')


def record_results(batch_datapoints, batch_p_answers):
    global new_datapoints
    temp = []
    for i, datapoint in enumerate(batch_datapoints):
        datapoint['predicted_answer'] = batch_p_answers[i]
        temp.append(datapoint)
    new_datapoints += temp


def evaluate_on_validation_dataset(tester, considered_group=None):
    hits = []
    counter = 0
    batch_texts = []
    batch_answers = []
    batch_p_answers = []
    batch_datapoints = []
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            if considered_group is None:
                pass
            else:
                if datapoint['group'] != considered_group:
                    continue
            text = train_text(datapoint)
            answer = train_label(datapoint)
            batch_texts.append(text)
            batch_answers.append(answer)
            batch_datapoints.append(datapoint)
            if len(batch_texts) == configuration.batch_size:
                predictions = tester.evaluate(batch_texts)
                for i, prediction in enumerate(predictions):
                    p_answer = prediction.item()
                    answer = batch_answers[i]
                    batch_p_answers.append(p_answer)
                    if answer == p_answer:
                        hits.append(1)
                    else:
                        hits.append(0)
                counter += len(batch_texts)
                record_results(batch_datapoints, batch_p_answers)
                batch_p_answers = []
                batch_datapoints = []
                batch_texts = []
                batch_answers = []
                logging.info(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
                print(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
    if batch_texts:
        predictions = tester.evaluate(batch_texts)
        for i, prediction in enumerate(predictions):
            p_answer = prediction.item()
            answer = batch_answers[i]
            batch_p_answers.append(p_answer)
            if answer == p_answer:
                hits.append(1)
            else:
                hits.append(0)
        counter += len(batch_texts)
        record_results(batch_datapoints, batch_p_answers)
        logging.info(f"Final batch processed. Total processed {counter} datapoints.")
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}% for {considered_group}"
    logging.info(text)
    print(text)
    write_results()


if __name__ == '__main__':
    initialization()
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # evaluate_on_validation_dataset(obj)

    # groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
    #           'bitterest food items', 'umami food items', 'fattiest food items']

    # groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
    #           'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
    #           'densest rocks']
    # for group in groups:
    #     evaluate_on_validation_dataset(obj, group)

    # This configuration if for generating Table 2 results
    # ====================================================
    # configuration.base_model_id = "meta-llama/Llama-2-7b-hf"
    # logging.info(f'Running base model = {configuration.base_model_id}')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj
    #
    # configuration.base_model_id = "meta-llama/Llama-2-13b-hf"
    # logging.info(f'Running base model = {configuration.base_model_id}')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj
    #
    # configuration.base_model_id = "mistralai/Mistral-7B-v0.1"
    # logging.info(f'Running base model = {configuration.base_model_id}')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"llm_training_wikidata.jsonl")
    # configuration.max_steps = 15000
    # logging.info(f'Trained on training data = {configuration.training_dataset}, WD')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"llm_training_books_movies.jsonl")
    # configuration.max_steps = 15000
    # logging.info(f'Trained on training data = {configuration.training_dataset}, TG')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"llm_training_foods.jsonl")
    # configuration.max_steps = 15000
    # logging.info(f'Trained on training data = {configuration.training_dataset}, Taste')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"llm_training_books_movies_wikidata_foods.jsonl")
    # configuration.max_steps = 25000
    # logging.info(f'Trained on training data = {configuration.training_dataset}, WD+TG+Taste')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                   f"train_validate/llm_training_wikidata_main.jsonl")
    # configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    # configuration.max_steps = 12500
    # logging.info(f'Trained on training data = {configuration.training_dataset}, WD1-train')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_wikidata_main_500_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_wikidata_subtle_500_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_food_taste_500_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
    #           'bitterest food items', 'umami food items', 'fattiest food items']
    # for group in groups:
    #     evaluate_on_validation_dataset(obj, group)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_rocks_500_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
    #           'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
    #           'densest rocks']
    # for group in groups:
    #     evaluate_on_validation_dataset(obj, group)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_books_100_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
    #                                     f"train_validate/llm_validation_new_movies_100_datapoints.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    # configuration.training_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"llm_training_wikidata_foods_rocks.jsonl")
    # configuration.max_steps = 25000
    # logging.info(f'Trained on training data = {configuration.training_dataset}, WD+Taste+Rocks')
    # obj = EvalLargeLanguageModel()
    # obj.init_ft_model()
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"sizes_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"heights_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    #
    # configuration.validation_dataset = (
    #     f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
    #     f"masses_validation.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # evaluate_on_validation_dataset(obj)
    # logging.info(f'Validation file evaluated is {configuration.validation_dataset}')
    # del obj

    configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    configuration.max_steps = 12500
    logging.info(f'Running base model = {configuration.base_model_id}')
    obj = EvalLargeLanguageModel()
    obj.init_ft_model()

    if sys.argv[1] == "WD1-train" or sys.argv[1] == "TG" or sys.argv[1] == "Taste":
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_wikidata_main_500_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if sys.argv[1] == "WD1-train" or sys.argv[1] == "TG" or sys.argv[1] == "Taste":
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_wikidata_subtle_500_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if sys.argv[1] == "WD1-train" or sys.argv[1] == "WD" or sys.argv[1] == "TG" or sys.argv[1] == "WD+TG+Rocks":
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_food_taste_500_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
                  'bitterest food items', 'umami food items', 'fattiest food items']
        for group in groups:
            evaluate_on_validation_dataset(obj, group)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if (sys.argv[1] == "WD1-train" or sys.argv[1] == "WD" or sys.argv[1] == "TG" or sys.argv[1] == "Taste" or
            sys.argv[1] == "WD+TG+Taste"):
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_rocks_500_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
                  'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
                  'densest rocks']
        for group in groups:
            evaluate_on_validation_dataset(obj, group)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if sys.argv[1] == "WD1-train" or sys.argv[1] == "WD" or sys.argv[1] == "Taste" or sys.argv[1] == "WD+Taste+Rocks":
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_movies_100_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if sys.argv[1] == "WD1-train" or sys.argv[1] == "WD" or sys.argv[1] == "Taste" or sys.argv[1] == "WD+Taste+Rocks":
        configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                            f"train_validate/llm_validation_new_books_100_datapoints.jsonl")
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)
        logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    if (sys.argv[1] == "WD1-train" or sys.argv[1] == "WD" or sys.argv[1] == "TG" or sys.argv[1] == "Taste"
            or sys.argv[1] == "WD+TG+Taste" or sys.argv[1] == "WD+TG+Rocks" or sys.argv[1] == "WD+Taste+Rocks"):
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
    del obj

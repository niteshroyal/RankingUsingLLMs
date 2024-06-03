import os
import json
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from llm_classifier.pointwise import LargeLanguageModelPointWise

from conf import configuration
from llm_classifier.prepare_dataset import pointwise_eval_text, pointwise_eval_label


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class EvalLargeLanguageModelPointWise(LargeLanguageModelPointWise):

    def __init__(self):
        super().__init__()
        self.fine_tuned_model_location = None
        self.ft_model = None
        self.init_conf()
        self.loss_name = 'bce'  # 'hinge'

    def init_ft_model(self):
        self.fine_tuned_model_location = self.get_generator_llm()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=1,
            quantization_config=self.bnb_config
        )
        self.ft_model = PeftModel.from_pretrained(base_model, self.fine_tuned_model_location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id

    def evaluate(self, eval_prompts, max_new_tokens=configuration.tokenizer_max_length):
        n = len(eval_prompts)
        all_prompts = []
        for [prompt1, _] in eval_prompts:
            all_prompts.append(prompt1)
        for [_, prompt2] in eval_prompts:
            all_prompts.append(prompt2)
        model_input = self.tokenizer(
            all_prompts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            eval_logits = eval_outputs.logits
            scores1 = eval_logits[:n]
            scores2 = eval_logits[n:]
            prediction = (scores1 > scores2).float().squeeze()
            return prediction


def evaluate_on_validation_dataset(tester, considered_group=None):
    hits = []
    counter = 0
    batch_texts = []
    batch_answers = []
    assert configuration.batch_size % 2 == 0, "batch_size is not even"
    validation_batch_size = int(configuration.batch_size / 2)
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            if considered_group is None:
                pass
            else:
                if datapoint['group'] != considered_group:
                    continue
            [text1, text2] = pointwise_eval_text(datapoint)
            answer = pointwise_eval_label(datapoint)
            batch_texts.append([text1, text2])
            batch_answers.append(answer)
            if len(batch_texts) == validation_batch_size:
                predictions = tester.evaluate(batch_texts)
                for i, prediction in enumerate(predictions):
                    p_answer = prediction.item()
                    answer = batch_answers[i]
                    if answer == p_answer:
                        hits.append(1)
                    else:
                        hits.append(0)
                counter += len(batch_texts)
                batch_texts = []
                batch_answers = []
                logging.info(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
                print(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
    if batch_texts:
        predictions = tester.evaluate(batch_texts)
        for i, prediction in enumerate(predictions):
            p_answer = prediction.item()
            answer = batch_answers[i]
            if answer == p_answer:
                hits.append(1)
            else:
                hits.append(0)
        counter += len(batch_texts)
        logging.info(f"Final batch processed. Total processed {counter} datapoints.")
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}% for {considered_group}"
    logging.info(text)
    print(text)


if __name__ == '__main__':
    initialization()
    # obj = EvalLargeLanguageModelPointWise()
    # obj.init_ft_model()

    # evaluate_on_validation_dataset(obj)

    # groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
    #           'bitterest food items', 'umami food items', 'fattiest food items']

    # groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
    #           'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
    #           'densest rocks']
    # for group in groups:
    #     evaluate_on_validation_dataset(obj, group)

    configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    configuration.max_steps = 12500
    logging.info(f'Running base model = {configuration.base_model_id}')
    obj = EvalLargeLanguageModelPointWise()
    obj.init_ft_model()

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_wikidata_main_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_wikidata_subtle_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_food_taste_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ['sweetest food items', 'saltiest food items', 'sourest food items',
              'bitterest food items', 'umami food items', 'fattiest food items']
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_rocks_500_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ['lightest-colored rocks', 'coarsest rocks', 'roughest rocks', 'shiniest rocks',
              'rocks with the most uniform grain structure', 'rocks with the greatest variability in color',
              'densest rocks']
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_books_100_datapoints.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    evaluate_on_validation_dataset(obj)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    configuration.validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
                                        f"train_validate/llm_validation_new_movies_100_datapoints.jsonl")
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
    del obj

    # configuration.base_model_id = "meta-llama/Llama-2-13b-hf"
    # logging.info(f'Running base model = {configuration.base_model_id}')
    # obj = EvalLargeLanguageModelPointWise()
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
    # obj = EvalLargeLanguageModelPointWise()
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

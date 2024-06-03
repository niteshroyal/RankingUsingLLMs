import os
import torch
import logging

from transformers import AutoModelForSequenceClassification
from llama_finetuning.llm import get_generator_llm
from llm_classifier.bert_classifier import BERTLanguageModel
from llm_classifier.eval_classifier import evaluate_on_validation_dataset

from conf import configuration


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class EvalBERTLanguageModel(BERTLanguageModel):
    def __init__(self):
        super().__init__()
        self.fine_tuned_model_location = None
        self.ft_model = None
        self.init_conf()

    def init_ft_model(self):
        self.fine_tuned_model_location = get_generator_llm()
        self.ft_model = AutoModelForSequenceClassification.from_pretrained(
            self.fine_tuned_model_location,
            num_labels=2,
        )

    def evaluate(self, eval_prompts, max_new_tokens=configuration.tokenizer_max_length):
        model_input = self.tokenizer(
            eval_prompts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        )

        model_input = {k: v.to(self.ft_model.device) for k, v in model_input.items()}

        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            eval_logits = eval_outputs.logits
            eval_predictions = torch.argmax(eval_logits, dim=1)

        return eval_predictions


if __name__ == '__main__':
    initialization()

    configuration.training_dataset = (
        f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
        f"llm_training_books_movies_wikidata_rocks.jsonl")
    configuration.max_steps = 25000
    logging.info(f'Trained on training data = {configuration.training_dataset}, WD+TG+Rocks')
    obj = EvalBERTLanguageModel()
    obj.init_ft_model()

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

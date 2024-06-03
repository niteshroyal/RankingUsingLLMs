import os
import torch
import logging
import transformers

from conf import configuration
from llm_classifier.pointwise import (LargeLanguageModelPointWise, TrainDatasetLoader, EvalDatasetLoader,
                                      train_dataset, eval_dataset, get_old_generator_llm_location)


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class TrainDatasetLoaderNormalizedRank(TrainDatasetLoader):

    def __init__(self, datapoints, tokenizer):
        super().__init__(datapoints, tokenizer)

    def normalize_ranks(self):
        for group in self.groups:
            ranks = []
            for i, datapoint in enumerate(self.processed_datapoints[group]):
                ranks.append(datapoint['labels'].item())
            min_ranks = min(ranks)
            range_ranks = max(ranks) - min_ranks
            normalized_ranks = [(x - min_ranks) / range_ranks for x in ranks]
            for i, datapoint in enumerate(self.processed_datapoints[group]):
                datapoint['labels'] = torch.tensor(normalized_ranks[i], dtype=torch.float)


class LargeLanguageModelPointWiseHingeLoss(LargeLanguageModelPointWise):

    def __init__(self):
        super().__init__()

    def init_trainer(self):
        self.loss_name = 'hinge'
        self.training_data_loader = TrainDatasetLoaderNormalizedRank(train_dataset, self.tokenizer)
        self.training_data_loader.process()
        self.training_data_loader.normalize_ranks()
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()
        self.trainer = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                output_dir=get_old_generator_llm_location()
            )
        )


if __name__ == '__main__':
    initialization()
    obj = LargeLanguageModelPointWiseHingeLoss()
    obj.init_conf()
    obj.init_model()
    obj.init_trainer()
    # obj.determine_tokenizer_max_length()
    obj.finetuning()
    obj.rename_finetuned_model_path()

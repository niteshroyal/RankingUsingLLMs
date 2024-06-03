import os
import json
import random
import logging

from conf import configuration
from utils.utils import read_data
from analysis.prompts_pointwise import (get_wikidata_prompts, get_prompt_taste,
                                        get_prompt_rocks, get_prompt_books, get_prompt_movies,
                                        get_prompt_physical_properties)


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_prompt(record1, record2, filename):
    return get_prompt_physical_properties(record1, record2, filename)
    # return get_prompt_rocks(record1, record2, filename)
    # return get_prompt_taste(record1, record2, filename)
    # return get_prompt2(record1, record2, filename)
    # return [get_wikidata_prompts(record1, record2, filename)]
    # return get_prompt_books(record1, record2, filename)
    # return get_prompt_movies(record1, record2, filename)


def get_random_indices(n, m):
    all_indices = []
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                pass
            else:
                all_indices.append((i, j))
    size_all_indices = len(all_indices)
    if size_all_indices <= m:
        random.shuffle(all_indices)
        return all_indices
    else:
        sampled_indices = random.sample(all_indices, m)
        return sampled_indices


class Controller:

    def __init__(self):
        self.training_datapoints = []
        self.validation_datapoints = []
        random.seed(42)

    def dataset_for_fine_tuning(self, data, filename):
        topn = configuration.topn
        num_of_entities = min(topn, len(data))
        topn_training = round(configuration.percentage_of_training_dataset * num_of_entities)
        topn_validation = num_of_entities - topn_training
        selected_data = data[0:num_of_entities]
        random.shuffle(selected_data)
        training_selected_data = selected_data[0:topn_training]
        validation_selected_data = selected_data[topn_training:num_of_entities]
        training_indices = get_random_indices(topn_training, configuration.num_training_datapoints)
        for (idx, idx2) in training_indices:
            record1 = training_selected_data[idx]
            record2 = training_selected_data[idx2]
            datapoints = get_prompt(record1, record2, filename)
            for datapoint in datapoints:
                self.training_datapoints.append(datapoint)
        validation_indices = get_random_indices(topn_validation, configuration.num_validation_datapoints)
        for (idx, idx2) in validation_indices:
            record1 = validation_selected_data[idx]
            record2 = validation_selected_data[idx2]
            datapoints = get_prompt(record1, record2, filename)
            for datapoint in datapoints:
                self.validation_datapoints.append(datapoint)

    def get_training_datapoints_for_pointwise_approach(self):
        training_datapoints_for_pointwise_approach = []
        groups = set()
        for item in self.training_datapoints:
            groups.add(item['group'])
        groups = list(groups)
        sorted_scores_groupwise = dict()
        for group in groups:
            sorted_scores_groupwise[group] = set()
        for item in self.training_datapoints:
            sorted_scores_groupwise[item['group']].add(item['score1'])
            sorted_scores_groupwise[item['group']].add(item['score2'])
        for key in sorted_scores_groupwise:
            sorted_scores_groupwise[key] = list(sorted_scores_groupwise[key])
            sorted_scores_groupwise[key].sort(reverse=True)
        prompt_already_entered = dict()
        for group in groups:
            prompt_already_entered[group] = []
        for item in self.training_datapoints:
            group = item['group']
            score1 = item['score1']
            score2 = item['score2']
            rank1 = sorted_scores_groupwise[group].index(score1) + 1
            rank2 = sorted_scores_groupwise[group].index(score2) + 1
            prompt1 = item['prompt1']
            prompt2 = item['prompt2']
            if prompt1 not in prompt_already_entered[group]:
                datapoint = dict()
                datapoint['group'] = group
                datapoint['question'] = prompt1
                datapoint['rank'] = rank1
                training_datapoints_for_pointwise_approach.append(datapoint)
                prompt_already_entered[group].append(prompt1)
            if prompt2 not in prompt_already_entered[group]:
                datapoint = dict()
                datapoint['group'] = group
                datapoint['question'] = prompt2
                datapoint['rank'] = rank2
                training_datapoints_for_pointwise_approach.append(datapoint)
                prompt_already_entered[group].append(prompt2)
        return training_datapoints_for_pointwise_approach

    def get_validation_datapoints_for_pointwise_approach(self):
        validation_datapoints_for_pointwise_approach = []
        for item in self.validation_datapoints:
            datapoint = dict()
            datapoint['1_rank_higher_than_2'] = item['answer']
            datapoint['prompt1'] = item['prompt1']
            datapoint['prompt2'] = item['prompt2']
            datapoint['score1'] = item['score1']
            datapoint['score2'] = item['score2']
            datapoint['answer'] = item['answer']
            datapoint['group'] = item['group']
            datapoint['question'] = item['question']
            datapoint['element1'] = item['element1']
            datapoint['element2'] = item['element2']
            validation_datapoints_for_pointwise_approach.append(datapoint)
        return validation_datapoints_for_pointwise_approach

    def write_files(self):
        random.shuffle(self.training_datapoints)
        training_datapoints_for_pointwise_approach = self.get_training_datapoints_for_pointwise_approach()
        with open(configuration.training_dataset, 'w', encoding='utf-8') as file_handler:
            for item in training_datapoints_for_pointwise_approach:
                json.dump(item, file_handler)
                file_handler.write('\n')
        random.shuffle(self.validation_datapoints)
        validation_datapoints_for_pointwise_approach = self.get_validation_datapoints_for_pointwise_approach()
        with open(configuration.validation_dataset, 'w', encoding='utf-8') as file_handler:
            for item in validation_datapoints_for_pointwise_approach:
                json.dump(item, file_handler)
                file_handler.write('\n')

    def control(self):
        all_files = os.listdir(configuration.ranking_dataset_folder)
        txt_files = [file for file in all_files if file.endswith('.txt')]
        for file_name in txt_files:
            file_path = os.path.join(configuration.ranking_dataset_folder, file_name)
            headers, data = read_data(file_path)
            self.dataset_for_fine_tuning(data, file_name)
        self.write_files()


if __name__ == '__main__':
    initialization()
    obj = Controller()
    obj.control()

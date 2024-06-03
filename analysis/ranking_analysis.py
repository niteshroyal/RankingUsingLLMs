import os
import csv
import torch
import random
import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.svm import SVC

from conf import configuration
from llm_classifier.eval_pointwise import EvalLargeLanguageModelPointWise
from llm_classifier.eval_classifier import EvalLargeLanguageModel
from utils.utils import read_data


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def spearman_rho(original_rank, predicted_rank):
    # Ensure the lists are of the same length
    if len(original_rank) != len(predicted_rank):
        raise ValueError("Lists must be of the same length")

    n = len(original_rank)
    # Calculate the squared differences between each pair of ranks
    d_squared = [(o - p) ** 2 for o, p in zip(original_rank, predicted_rank)]
    sum_d_squared = sum(d_squared)

    # Calculate Spearman's rho
    rho = 1 - (6 * sum_d_squared) / (n * (n ** 2 - 1))
    return rho


def calculate_ranks(score_dict):
    # Convert score_dict to a list of tuples and sort by score in descending order
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    # Create a temporary list to store (name, score, rank)
    temp = []
    for i, (name, score) in enumerate(sorted_scores):
        # Rank is initially the position in the sorted list + 1 (because index starts at 0)
        temp.append((name, score, i + 1))

    # Handle ties by assigning the average rank for tied scores
    i = 0
    while i < len(temp):
        score = temp[i][1]
        start = i
        while i + 1 < len(temp) and temp[i + 1][1] == score:
            i += 1
        end = i

        # Compute the average rank for the tied scores
        avg_rank = sum(temp[j][2] for j in range(start, end + 1)) / (end - start + 1)

        # Assign the average rank to all tied scores
        for j in range(start, end + 1):
            temp[j] = (temp[j][0], temp[j][1], avg_rank)

        i += 1

    # Convert the temp list back to a dictionary
    rank_dict = {name: rank for name, score, rank in temp}
    return rank_dict


def select_n_items_exclude_x(items, n, x):
    # Filter out the item x from the list
    filtered_items = [item for item in items if item != x]

    # Check if we have enough items left to select n items
    if len(filtered_items) < n:
        raise ValueError("Not enough items in the list after excluding x to select n items.")

    # Randomly select n items from the filtered list
    selected_items = random.sample(filtered_items, n)

    return selected_items


class Taste:
    def __init__(self, prop=None, item_label=None):
        self.data_filename = configuration.qualitative_analysis_dataset
        self.headers, data = read_data(self.data_filename)
        num_of_entities = min(configuration.topn, len(data))
        self.data = data[0:num_of_entities]
        self.prop = prop
        self.item_label = item_label
        self.prompts = []
        np.random.seed(42)

    def set_prop(self, prop=None):
        self.prop = prop

    def set_item_label(self, item_label=None):
        self.item_label = item_label

    def add_prompts_for_pointwise(self):
        for datapoint in self.data:
            datapoint_dict = dict()
            datapoint_dict['item'] = datapoint[self.item_label]
            datapoint_dict['original_score'] = float(datapoint[self.prop])
            if self.prop == 'Sweet_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the sweetest food items?'
            elif self.prop == 'Salty_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the saltiest food items?'
            elif self.prop == 'Sour_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the sourest food items?'
            elif self.prop == 'Bitter_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the bitterest food items?'
            elif self.prop == 'Umami_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the most umami food items?'
            elif self.prop == 'Fat_Mean':
                datapoint_dict['pointwise_prompt'] = f'Is {datapoint[self.item_label]} among the fattiest food items?'
            else:
                raise Exception('Unknown property')
            self.prompts.append(datapoint_dict)

    def add_prompts_for_pairwise(self, num_choices):
        n = len(self.data)
        indices = list(range(0, n))
        for i, datapoint in enumerate(self.data):
            datapoint_dict = dict()
            comparisons = []
            comparison_prompts = []
            choices = select_n_items_exclude_x(indices, num_choices, i)
            for choice in choices:
                comparisons.append([i, choice])
                prompt = self.get_pairwise_prompt(self.data[i][self.item_label], self.data[choice][self.item_label])
                comparison_prompts.append(prompt)
            datapoint_dict['item'] = datapoint[self.item_label]
            datapoint_dict['original_score'] = float(datapoint[self.prop])
            datapoint_dict['comparison_indices'] = comparisons
            datapoint_dict['comparison_prompts'] = comparison_prompts
            self.prompts.append(datapoint_dict)

    def add_original_ranks(self):
        score_dict = dict()
        for prompt in self.prompts:
            score_dict[prompt['item']] = prompt['original_score']
        rank_dict = calculate_ranks(score_dict)
        for prompt in self.prompts:
            prompt['original_rank'] = rank_dict[prompt['item']]

    def add_predicted_ranks(self):
        score_dict = dict()
        for prompt in self.prompts:
            score_dict[prompt['item']] = prompt['predicted_score']
        rank_dict = calculate_ranks(score_dict)
        for prompt in self.prompts:
            prompt['predicted_rank'] = rank_dict[prompt['item']]

    def get_pairwise_prompt(self, element1, element2):
        if self.prop == 'Sweet_Mean':
            question = f'This question is about two food items: Is {element1} generally sweeter in taste than {element2}?'
        elif self.prop == 'Salty_Mean':
            question = f'This question is about two food items: Is {element1} generally saltier than {element2}?'
        elif self.prop == 'Sour_Mean':
            question = f'This question is about two food items: Is {element1} generally more sour in taste than {element2}?'
        elif self.prop == 'Bitter_Mean':
            question = f'This question is about two food items: Is {element1} generally more bitter in taste than {element2}?'
        elif self.prop == 'Umami_Mean':
            question = f'This question is about two food items: Is {element1} generally more umami than {element2}?'
        elif self.prop == 'Fat_Mean':
            question = f'This question is about two food items: Does {element1} taste fattier than {element2}?'
        else:
            raise Exception('Unknown property')
        return question

    def dump_prompts(self):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]

        dump_file_for_qualitative_analysis = ('/scratch/c.scmnk4/elexir/ranking_research/resources'
                                              f'/Ranking_DataSet/qualitative_analysis_of_{filename}_{self.prop}.csv')
        # with open(dump_file_for_qualitative_analysis, 'w', encoding='utf-8') as file_handler:
        #     for item in self.prompts:
        #         json.dump(item, file_handler)
        #         file_handler.write('\n')
        items = []
        original_scores = []
        original_ranks = []
        predicted_scores = []
        predicted_ranks = []
        for item in self.prompts:
            items.append(item['item'])
            original_scores.append(item['original_score'])
            original_ranks.append(item['original_rank'])
            predicted_scores.append(item['predicted_score'])
            predicted_ranks.append(item['predicted_rank'])
        with open(dump_file_for_qualitative_analysis, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["item", "original_score", "original_rank", "predicted_score", "predicted_rank"])
            for a, b, c, d, e in zip(items, original_scores, original_ranks, predicted_scores, predicted_ranks):
                writer.writerow([a, b, c, d, e])


class Movies(Taste):

    def __init__(self, prop=None):
        super().__init__(prop)

    def get_pairwise_prompt(self, element1, element2):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]
        if filename == 'NEWFunny_movie_titles1':
            question = f'This question is about two movies: Is {element1} more funny than {element2}?'
        elif filename == 'NEWScary_movie_titles1':
            question = f'This question is about two movies: Is {element1} more scary than {element2}?'
        else:
            raise Exception('Unknown property')
        return question


class WikiData(Taste):
    def __init__(self, prop=None):
        super().__init__(prop)

    def get_pairwise_prompt(self, element1, element2):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]
        if filename == 'matched_query_rankCountries_Population':
            question = f'This question is about two countries: Does {element1} have a larger population than {element2}?'
        elif filename == 'matched_query_Rank_BuildingHeight_WikiRank':
            question = f'This question is about two buildings: Is {element1} taller than {element2}?'
        else:
            raise Exception('Unknown property')
        return question

    def set_prop(self, prop=None):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]
        if filename == 'matched_query_rankCountries_Population':
            self.prop = 'maxPopulation'
        elif filename == 'matched_query_Rank_BuildingHeight_WikiRank':
            self.prop = 'maxHeight'
        else:
            raise Exception('Unknown property')

    def set_item_label(self, item_label=None):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]
        if filename == 'matched_query_rankCountries_Population':
            self.item_label = 'countryLabel'
        elif filename == 'matched_query_Rank_BuildingHeight_WikiRank':
            self.item_label = 'itemLabel'
        else:
            raise Exception('Unknown property')


class Extras(Taste):
    def __init__(self, prop=None):
        super().__init__(prop)

    def get_pairwise_prompt(self, element1, element2):
        filename = os.path.splitext(os.path.basename(configuration.qualitative_analysis_dataset))[0]
        if filename == 'present_for_10_year_old_girl':
            question = (f'This question is about two objects: Is {element1} more suitable as a present for a '
                        f'10-year-old girl than {element2}?')
        elif filename == 'urgent_medical_treatment':
            question = (f'This question is about two medical conditions: Does {element1} require more urgent '
                        f'medical treatment than {element2}?')
        elif filename == 'cheaper_to_live':
            question = f'This question is about two cities: Is {element1} cheaper to live than {element2}?'
        elif filename == 'partying_post_marathon':
            question = (f'This question is about two activities: Is {element1} more important than {element2} '
                        f'post marathon?')
        elif filename == 'more_calories':
            question = f'This question is about two routines: Does {element1} burn more calories than {element2}?'
        else:
            raise Exception('Unknown property')
        return question


def SVMRanking(dataset_obj, all_comparison_indices, all_predictions):
    comparisons = []
    labels = []
    indicator_vectors = np.eye(len(dataset_obj.data))

    for [idx1, idx2], prediction in zip(all_comparison_indices, all_predictions):
        comparison_vector = indicator_vectors[idx1] - indicator_vectors[idx2]
        comparisons.append(comparison_vector)
        labels.append(prediction)

    comparisons = np.array(comparisons)
    labels = np.array(labels)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(comparisons, labels)

    w = svm.coef_[0]
    for i, prompt in enumerate(dataset_obj.prompts):
        prompt['predicted_score'] = w[i]


def RankingByCount(dataset_obj, all_comparison_indices, all_predictions):
    n = len(dataset_obj.prompts)
    won = np.zeros(n)
    contested = np.zeros(n)
    for [idx1, idx2], prediction in zip(all_comparison_indices, all_predictions):
        if prediction == 1:
            won[idx1] += 1
        else:
            won[idx2] += 1
        contested[idx1] += 1
        contested[idx2] += 1
    score = won / contested
    for i, prompt in enumerate(dataset_obj.prompts):
        prompt['predicted_score'] = score[i]


class Pairwise(EvalLargeLanguageModel):

    def __init__(self):
        super().__init__()
        self.init_ft_model()

    def add_predicted_scores(self, dataset_obj, approach):
        assert len(dataset_obj.data) == len(dataset_obj.prompts)

        all_comparison_indices = []
        all_comparison_prompts = []
        all_predictions = []
        for prompt in dataset_obj.prompts:
            all_comparison_indices += prompt['comparison_indices']
            all_comparison_prompts += prompt['comparison_prompts']
        batch_texts = []
        counter = 0
        for prompt in all_comparison_prompts:
            batch_texts.append(prompt)
            if len(batch_texts) == configuration.batch_size:
                predictions = self.evaluate(batch_texts)
                for i, prediction in enumerate(predictions):
                    predicted_score = prediction.item()
                    result = 1 if predicted_score == 1 else -1
                    all_predictions.append(result)
                counter += len(batch_texts)
                batch_texts = []
                logging.info(f"Pairwise model accessed for {counter} datapoints")
                print(f"Pairwise model accessed for {counter} datapoints")
        if batch_texts:
            predictions = self.evaluate(batch_texts)
            for i, prediction in enumerate(predictions):
                predicted_score = prediction.item()
                result = 1 if predicted_score == 1 else -1
                all_predictions.append(result)
            counter += len(batch_texts)
            logging.info(f"Pairwise model accessed for {counter} datapoints")
            print(f"Pairwise model accessed for {counter} datapoints")
        if approach == 'svm':
            SVMRanking(dataset_obj, all_comparison_indices, all_predictions)
        elif approach == 'count':
            RankingByCount(dataset_obj, all_comparison_indices, all_predictions)
        else:
            raise Exception('Incorrect approach')


class Pointwise(EvalLargeLanguageModelPointWise):

    def __init__(self):
        super().__init__()
        self.init_ft_model()

    def evaluate(self, texts, max_new_tokens=configuration.tokenizer_max_length):
        model_input = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            eval_logits = eval_outputs.logits
            return eval_logits

    def add_predicted_scores(self, dataset_obj):
        batch_texts = []
        batch_prompt = []
        counter = 0
        for prompt in dataset_obj.prompts:
            batch_prompt.append(prompt)
            batch_texts.append(prompt['pointwise_prompt'])
            if len(batch_texts) == configuration.batch_size:
                predictions = self.evaluate(batch_texts)
                for i, prediction in enumerate(predictions):
                    predicted_score = prediction.item()
                    temp_prompt = batch_prompt[i]
                    temp_prompt['predicted_score'] = predicted_score
                counter += len(batch_texts)
                batch_texts = []
                batch_prompt = []
                logging.info(f"Pointwise model accessed for {counter} datapoints")
                print(f"Pointwise model accessed for {counter} datapoints")
        if batch_texts:
            predictions = self.evaluate(batch_texts)
            for i, prediction in enumerate(predictions):
                predicted_score = prediction.item()
                temp_prompt = batch_prompt[i]
                temp_prompt['predicted_score'] = predicted_score
            counter += len(batch_texts)
            logging.info(f"Pointwise model accessed for {counter} datapoints")
            print(f"Pointwise model accessed for {counter} datapoints")


class ItemsRanking:

    def __init__(self):
        self.dataset = None
        self.ranking_approach = None

    def set_ranking_approach(self, approach):
        if approach == 'pointwise':
            self.ranking_approach = Pointwise()
        elif approach == 'pairwise':
            self.ranking_approach = Pairwise()
        else:
            raise Exception('Incorrect ranking approach')

    def calculate_spearman_rho(self):
        original = []
        predicted = []
        for prompt in self.dataset.prompts:
            original.append(prompt['original_rank'])
            predicted.append(prompt['predicted_rank'])
        # rho = spearman_rho(original, predicted)
        res = spearmanr(np.array(original), np.array(predicted))
        rho = res.correlation
        logging.info(f'Spearman rho of {self.dataset.prop} is {rho}')
        print(f'Spearman rho of {self.dataset.prop} is {rho}')

    def pointwise_control(self, prop=None):
        # List of properties for Taste dataset
        # ['Sweet_Mean', 'Salty_Mean', 'Sour_Mean', 'Bitter_Mean', 'Umami_Mean', 'Fat_Mean']

        self.dataset = Taste()
        self.dataset.set_item_label("foodLabel")
        self.dataset.set_prop(prop)

        self.dataset.add_prompts_for_pointwise()
        self.dataset.add_original_ranks()
        self.ranking_approach.add_predicted_scores(self.dataset)
        self.dataset.add_predicted_ranks()
        self.calculate_spearman_rho()

    def pairwise_control(self, prop=None, num_choices=None, scoring=None):
        # List of properties for Taste dataset
        # ['Sweet_Mean', 'Salty_Mean', 'Sour_Mean', 'Bitter_Mean', 'Umami_Mean', 'Fat_Mean']

        # self.dataset = Taste()
        # self.dataset.set_item_label("foodLabel")
        # self.dataset.set_prop(prop)

        # self.dataset = Movies()
        # self.dataset.set_item_label("title")
        # self.dataset.set_prop('score')

        # self.dataset = WikiData()
        # self.dataset.set_item_label()
        # self.dataset.set_prop()

        self.dataset = Extras()
        self.dataset.set_item_label("item")
        self.dataset.set_prop('score')

        self.dataset.add_prompts_for_pairwise(num_choices=num_choices)
        self.dataset.add_original_ranks()
        self.ranking_approach.add_predicted_scores(self.dataset, scoring)
        self.dataset.add_predicted_ranks()
        self.calculate_spearman_rho()
        self.dataset.dump_prompts()


if __name__ == '__main__':
    initialization()

    obj = ItemsRanking()

    # obj.set_ranking_approach('pointwise')
    #
    # properties = ['Sweet_Mean', 'Salty_Mean', 'Sour_Mean', 'Bitter_Mean', 'Umami_Mean', 'Fat_Mean']
    #
    # for p in properties:
    #     logging.info(f'Property is {p}')
    #     print(f'Property is {p}')
    #     obj.pointwise_control(p)

    obj.set_ranking_approach('pairwise')
    obj.pairwise_control(None, 5, 'svm')

    # logging.info('Going to use pairwise approach with count')
    # print('Going to use pairwise approach with count')
    # properties = ['Sweet_Mean', 'Salty_Mean', 'Sour_Mean', 'Bitter_Mean', 'Umami_Mean', 'Fat_Mean']
    # # properties = ['Sweet_Mean', 'Salty_Mean']
    # choices = [5, 30]
    # # choices = [30]
    # scoring_approaches = ['svm', 'count']
    # # scoring_approaches = ['svm']
    #
    # for choice in choices:
    #     for scoring_approach in scoring_approaches:
    #         for p in properties:
    #             logging.info(f'Property is {p}, Scoring approach is {scoring_approach}, and Number of Samples is {choice}')
    #             print(f'Property is {p}, Scoring approach is {scoring_approach}, and Number of Samples is {choice}')
    #             obj.pairwise_control(p, choice, scoring_approach)
    logging.info('Done')
    print('Done')

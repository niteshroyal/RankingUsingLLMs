import json
import random

from llm_classifier.pointwise import LargeLanguageModelPointWise

file_names = [
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_pointwise_books_genres_1000_datapoints.jsonl',
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_pointwise_movies_genres_1000_datapoints.jsonl',
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_pointwise_wikidata_main_subtle_1000_datapoints.jsonl',
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_pointwise_rocks_properties_5000_datapoints.jsonl'
]

training_file_name = (
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate'
    '/llm_training_pointwise_wikidata_movies_books_rocks.jsonl')

max_token_size = 50
all_records = []
excluded_records = []

obj = LargeLanguageModelPointWise()
obj.init_conf()

for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            prompt = json.loads(line)
            tokens = obj.generate_and_tokenize_prompt(prompt)
            tokens_size = len(tokens['input_ids'])
            if tokens_size < max_token_size:
                all_records.append(prompt)
            else:
                excluded_records.append(prompt)

random.shuffle(all_records)

print(f'Number of records less than max token length = {len(all_records)}')


with open(training_file_name, 'w', encoding='utf-8') as output_file:
    for record in all_records:
        output_file.write(json.dumps(record) + '\n')

print(
    f'Number of recorded datapoints = {len(all_records)}, Number of excluded datapoints = {len(excluded_records)}, '
    f'Number of all datapoints = {len(excluded_records) + len(all_records)}')
import json, random
from llm_classifier.classifier import LargeLanguageModelClassifier

file_names = [
    # '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    # '/llm_training_books_genres_1000_datapoints.jsonl',
    # '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    # '/llm_training_movies_genres_1000_datapoints.jsonl',
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_wikidata_main_subtle_properties_1000_datapoints.jsonl',
    # '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    # '/llm_training_foods_tastes_5000_datapoints.jsonl',
    # '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    # '/llm_training_wikidata_main_5000_datapoints.jsonl'
]

# file_names = [
#     '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
#     '/llm_validation_books_genres_100_datapoints.jsonl',
#     '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
#     '/llm_validation_movies_genres_100_datapoints.jsonl',
#
# ]

training_file_name = (
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate'
    '/llm_training_wikidata.jsonl')

# training_file_name = (
#     '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
#     '/llm_training_wikidata_main_subtle_top100_5000_datapoints.jsonl')

validation_file_name = (
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_validation_rocks_properties_500_datapoints.jsonl')

max_token_size = 50
all_records = []
excluded_records = []

obj = LargeLanguageModelClassifier()
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

yes_list = []
no_list = []
for rec in all_records:
    if rec["answer"] == "Yes":
        yes_list.append(rec)
    elif rec["answer"] == "No":
        no_list.append(rec)
    else:
        raise Exception()

max_len = min(len(yes_list), len(no_list))
new_all_records = yes_list[0:max_len] + no_list[0:max_len]
random.shuffle(new_all_records)

with open(training_file_name, 'w', encoding='utf-8') as output_file:
    for record in new_all_records:
        output_file.write(json.dumps(record) + '\n')

print(
    f'Number of recorded datapoints = {len(new_all_records)}, Number of excluded datapoints = {len(excluded_records)}, '
    f'Number of all datapoints = {len(excluded_records) + len(new_all_records)}')

tokens_size_list = []
with open(validation_file_name, 'r', encoding='utf-8') as file:
    for line in file:
        prompt = json.loads(line)
        tokens = obj.generate_and_tokenize_prompt(prompt)
        tokens_size = len(tokens['input_ids'])
        tokens_size_list.append(tokens_size)

print(f'max_token_size in {validation_file_name} is {max(tokens_size_list)}')

tokens_size_list = []
with open(training_file_name, 'r', encoding='utf-8') as file:
    for line in file:
        prompt = json.loads(line)
        tokens = obj.generate_and_tokenize_prompt(prompt)
        tokens_size = len(tokens['input_ids'])
        tokens_size_list.append(tokens_size)

print(f'max_token_size in {training_file_name} is {max(tokens_size_list)}')

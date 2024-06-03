import json

training_file_name = (
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_training_books_movies_taste.jsonl')

# training_file_name = (
#     '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/training_validation_files'
#     '/llm_training_food_taste_all_10000_datapoints.jsonl')

validation_file_name = (
    '/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files'
    '/llm_validation_rocks_all_500_datapoints.jsonl')

yes_list = []
no_list = []
with open(training_file_name, 'r', encoding='utf-8') as file:
    for line in file:
        answer = json.loads(line)["answer"]
        if answer == 'Yes':
            yes_list.append(1)
        elif answer == 'No':
            no_list.append(1)
        else:
            raise Exception()
print(f'In {training_file_name}, number of yes = {len(yes_list)}, number of no = {len(no_list)}')

yes_list = []
no_list = []
with open(validation_file_name, 'r', encoding='utf-8') as file:
    for line in file:
        answer = json.loads(line)["answer"]
        if answer == 'Yes':
            yes_list.append(1)
        elif answer == 'No':
            no_list.append(1)
        else:
            raise Exception()
print(f'In {validation_file_name}, number of yes = {len(yes_list)}, number of no = {len(no_list)}')

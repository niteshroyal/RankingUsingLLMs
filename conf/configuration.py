import os
import sys
import importlib.util

dir_path = os.path.dirname(os.path.realpath(__file__))

# if len(sys.argv) > 1:
#     if sys.argv[1] == "foods":
#         configuration_file_to_consider = os.path.join(dir_path, "hawk_conf_food.py")
#     elif sys.argv[1] == "movies_books":
#         configuration_file_to_consider = os.path.join(dir_path, "hawk_conf_movies_books.py")
#     elif sys.argv[1] == "rocks":
#         configuration_file_to_consider = os.path.join(dir_path, "hawk_conf_rocks.py")
#     else:
#         raise Exception('Appropriate arguments not passed to the code')
# else:
#     configuration_file_to_consider = os.path.join(dir_path, "my_conf.py")

configuration_file_to_consider = os.path.join(dir_path, "my_conf.py")


def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("conf", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module_from_file(configuration_file_to_consider)
logging_folder = config.logging_folder
learned_models = config.learned_models
qualitative_analysis_dataset = config.qualitative_analysis_dataset
topn = config.topn
ranking_dataset_folder = config.ranking_dataset_folder

percentage_of_training_dataset = config.percentage_of_training_dataset
num_training_datapoints = config.num_training_datapoints
num_validation_datapoints = config.num_validation_datapoints
entity_type = config.entity_type

training_dataset = config.training_dataset
validation_dataset = config.validation_dataset

base_model_id = config.base_model_id

load_in_kbit = config.load_in_kbit
lora_r = config.lora_r
lora_alpha = config.lora_alpha
tokenizer_max_length = config.tokenizer_max_length
max_steps = config.max_steps
save_steps = config.save_steps
eval_steps = config.eval_steps
max_new_tokens = config.max_new_tokens
batch_size = config.batch_size

server_url = config.server_url
run_id = config.run_id

# if len(sys.argv) > 1:
#     if sys.argv[1] == "meta-llama/Llama-2-7b-hf":
#         base_model_id = "meta-llama/Llama-2-7b-hf"
#     elif sys.argv[1] == "meta-llama/Llama-2-13b-hf":
#         base_model_id = "meta-llama/Llama-2-13b-hf"
#     elif sys.argv[1] == "mistralai/Mistral-7B-v0.1":
#         base_model_id = "mistralai/Mistral-7B-v0.1"
#     else:
#         raise Exception('Appropriate arguments not passed to the code')
# else:
#     pass


if len(sys.argv) > 1:
    if sys.argv[1] == "WD1-train":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_wikidata_main.jsonl")
    elif sys.argv[1] == "WD":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_wikidata.jsonl")
    elif sys.argv[1] == "TG":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_books_movies.jsonl")
    elif sys.argv[1] == "Taste":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_foods.jsonl")
    elif sys.argv[1] == "WD+TG+Taste":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_books_movies_wikidata_foods.jsonl")
    elif sys.argv[1] == "WD+TG+Rocks":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_books_movies_wikidata_rocks.jsonl")
    elif sys.argv[1] == "WD+Taste+Rocks":
        training_dataset = ("/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                            "/llm_training_wikidata_foods_rocks.jsonl")
    else:
        raise Exception('Appropriate arguments not passed to the code')
    run_id = int(sys.argv[2])
else:
    pass

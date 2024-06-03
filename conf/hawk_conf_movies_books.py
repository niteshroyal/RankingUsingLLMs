# General configuration
# ---------------------
logging_folder = "/home/c.scmnk4/elexir/RankingResearch/logs"
learned_models = "/scratch/c.scmnk4/elexir/ranking_research/learned_models"

# Configuration for analysis.controller.py
# ----------------------------------------
topn = 1000
ranking_dataset_folder = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet"
                          f"/Books")

percentage_of_training_dataset = 0.0
num_training_datapoints = 1000
num_validation_datapoints = 30

entity_type = 'wikidata_foods_rocks'

# Configuration for llama_finetuning.llm.py
# -----------------------------------------
training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                    f"/llm_training_wikidata_foods_rocks.jsonl")


# base_model_id = "meta-llama/Llama-2-7b-hf"
# base_model_id = "facebook/opt-350m"
# base_model_id = "facebook/bart-base"
# base_model_id = "mistralai/Mistral-7B-v0.1"
base_model_id = "meta-llama/Llama-2-13b-hf"
# base_model_id = "lmsys/vicuna-7b-v1.5"

load_in_kbit = 4
lora_r = 32
lora_alpha = 64
tokenizer_max_length = 50
max_steps = 25000
save_steps = max_steps
eval_steps = int(max_steps / 25)
batch_size = 8


# Configuration for llama_finetuning.client_llm.py
# ------------------------------------------------
validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate"
                      f"/llm_validation_movies_books_genres_100_datapoints.jsonl")


# server_url = 'http://10.98.36.251:5000/completion'
server_url = 'http://10.96.125.87:5000/completion'
max_new_tokens = 40

# General configuration
# ---------------------
logging_folder = "/home/nitesh/elexir/RankingResearch/logs"
learned_models = "/scratch/c.scmnk4/elexir/ranking_research/learned_models"

# Configuration for analysis.controller.py
# ----------------------------------------
# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Food_taste/food_Taste.txt')
# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Movie/NEWFunny_movie_titles1.txt')
# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Movie/NEWScary_movie_titles1.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'WikiDataMainProperties/matched_query_Rank_BuildingHeight_WikiRank.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'WikiDataSubtleProperties/matched_query_rankCountries_Population.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Extras/present_for_10_year_old_girl.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Extras/urgent_medical_treatment.txt')

qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
                                'Extras/cheaper_to_live.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Extras/more_calories.txt')

# qualitative_analysis_dataset = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/'
#                                 'Extras/partying_post_marathon.txt')


topn = 1000
ranking_dataset_folder = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet"
                          f"/PhysicalProperties/Mass")

percentage_of_training_dataset = 0.1
num_training_datapoints = 1000
num_validation_datapoints = 500

entity_type = 'objects_mass'

# Configuration for llama_finetuning.llm.py
# -----------------------------------------
# training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files/"
#                     f"llm_training_pointwise_{entity_type}_{num_training_datapoints}_datapoints.jsonl")
# training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
#                     f"llm_training_food_taste_all_10000_datapoints.jsonl")

# training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/test_datasets/"
#                     f"classifier_train.jsonl")

training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
                    f"llm_training_books_movies_wikidata_rocks.jsonl")

# training_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/test_datasets/"
#                     f"ranks_train.jsonl")

# base_model_id = "meta-llama/Llama-2-7b-hf"
# base_model_id = "google-bert/bert-large-uncased"
# base_model_id = "facebook/opt-350m"
# base_model_id = "facebook/bart-base"
# base_model_id = "mistralai/Mistral-7B-v0.1"
# base_model_id = "meta-llama/Llama-2-13b-hf"
# base_model_id = "lmsys/vicuna-7b-v1.5"
# base_model_id = "BlackSamorez/llama-2-tiny-testing"
# base_model_id = "microsoft/Phi-3-mini-4k-instruct"
base_model_id = "microsoft/phi-2"

load_in_kbit = 4
lora_r = 32
lora_alpha = 64
tokenizer_max_length = 50  # Determine this value by running llama_finetuning.determine_tokenizer_max_length.py
max_steps = 500
save_steps = max_steps
eval_steps = int(max_steps / 5)
batch_size = 8
run_id = 1

# Configuration for llama_finetuning.client_llm.py
# ------------------------------------------------
# validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/new_training_validation_files/"
#                       f"llm_validation_new_{entity_type}_{num_validation_datapoints}_datapoints.jsonl")

# validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
#                       f"llm_validation_rocks_all_500_datapoints.jsonl")

# validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/test_datasets/"
#                       f"classifier_eval.jsonl")

validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
                      f"llm_validation_new_food_taste_500_datapoints.jsonl")

# validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/test_datasets/"
#                       f"ranks_eval.jsonl")

# validation_dataset = (f"/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/train_validate/"
#                       f"llm_validation_new_rocks_500_datapoints.jsonl")


# server_url = 'http://10.98.36.251:5000/completion'
server_url = 'http://10.96.125.87:5000/completion'
max_new_tokens = 40

#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

BASE_PATH="/home/niteshkumar/elexir/RankingResearch"

cd $BASE_PATH

conda activate llm

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

if [ "$1" == "finetune_and_start_server" ]; then
    python $BASE_PATH/llm_classifier/classifier.py
    python $BASE_PATH/llm_classifier/server_llm_classifier.py
elif [ "$1" == "only_start_server" ]; then
    python $BASE_PATH/llm_classifier/server_llm_classifier.py
elif [ "$1" == "queue" ]; then
#    python $BASE_PATH/llm_classifier/classifier.py meta-llama/Llama-2-7b-hf >> $BASE_PATH/logs/classifier_queue 2>&1
#    python $BASE_PATH/llm_classifier/pointwise.py mistralai/Mistral-7B-v0.1 >> $BASE_PATH/logs/pointwise_queue.log 2>&1
#    python $BASE_PATH/llm_classifier/pointwise.py meta-llama/Llama-2-13b-hf >> $BASE_PATH/logs/pointwise_queue.log 2>&1

#    {
##      python $BASE_PATH/llm_classifier/classifier.py WD1-train 1
##      python $BASE_PATH/llm_classifier/classifier.py WD1-train 2
##      python $BASE_PATH/llm_classifier/classifier.py WD1-train 3
#
#      python $BASE_PATH/llm_classifier/classifier.py WD 1
##      python $BASE_PATH/llm_classifier/classifier.py WD 2
##      python $BASE_PATH/llm_classifier/classifier.py WD 3
#
#      python $BASE_PATH/llm_classifier/classifier.py TG 1
##      python $BASE_PATH/llm_classifier/classifier.py TG 2
##      python $BASE_PATH/llm_classifier/classifier.py TG 3
#
#      python $BASE_PATH/llm_classifier/classifier.py Taste 1
##      python $BASE_PATH/llm_classifier/classifier.py Taste 2
##      python $BASE_PATH/llm_classifier/classifier.py Taste 3
#
#      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Taste 1
##      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Taste 2
##      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Taste 3
#
#      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Rocks 1
##      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Rocks 2
##      python $BASE_PATH/llm_classifier/classifier.py WD+TG+Rocks 3
#
#      python $BASE_PATH/llm_classifier/classifier.py WD+Taste+Rocks 1
##      python $BASE_PATH/llm_classifier/classifier.py WD+Taste+Rocks 2
##      python $BASE_PATH/llm_classifier/classifier.py WD+Taste+Rocks 3
#    } >> $BASE_PATH/logs/classifier_queue_std.log 2>&1

#    {
##      python $BASE_PATH/llm_classifier/eval_classifier.py WD1-train 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py WD 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py TG 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py Taste 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py WD+TG+Taste 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py WD+TG+Rocks 1
#      python $BASE_PATH/llm_classifier/eval_classifier.py WD+Taste+Rocks 1
#    } >> $BASE_PATH/logs/eval_classifier_queue_std.log 2>&1

    {
      python $BASE_PATH/llm_classifier/eval_classifier.py WD+TG+Rocks 2
      python $BASE_PATH/llm_classifier/eval_classifier.py WD+TG+Rocks 3
      python $BASE_PATH/llm_classifier/eval_classifier.py WD+TG+Rocks 4
    } >> $BASE_PATH/logs/eval_classifier_queue_std.log 2>&1

#    python $BASE_PATH/llm_classifier/classifier.py foods >> $BASE_PATH/logs/classifier_queue 2>&1
#    python $BASE_PATH/llm_classifier/classifier.py books_movies >> $BASE_PATH/logs/classifier_queue 2>&1
#    python $BASE_PATH/llm_classifier/classifier.py wikidata >> $BASE_PATH/logs/classifier_queue 2>&1
else
    echo "Invalid argument. Please use 'finetune_and_start_server' or 'only_start_server' or 'queue'."
fi
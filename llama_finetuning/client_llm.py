import os
import json
import logging
import requests

from conf import configuration
from llama_finetuning.load_finetuning_dataset import formatting_func_for_eval


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_answer(response):
    parts = response.split("### Answer:")
    if len(parts) < 2:
        return "No answer section found."
    response_section = parts[1].split("### End")[0]
    return response_section.strip()


def evaluate_on_validation_dataset():
    hits = []
    counter = 0
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            text = formatting_func_for_eval(datapoint)
            data = {
                "prompt": text,
                "max_new_tokens": configuration.max_new_tokens
            }
            resp = requests.post(configuration.server_url, json=data)
            resp = resp.json()
            response = resp["llm_response"]
            answer = get_answer(response)
            logging.info(f"Answer: {datapoint['answer']}, Prediction: {answer}")
            if datapoint['answer'] == answer:
                hits.append(1)
            else:
                hits.append(0)
            counter += 1
            if counter % 10 == 0:
                text = (f"Number of validation datapoints processed = {counter}, "
                        f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
                logging.info(text)
                print(text)
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}%"
    logging.info(text)
    print(text)


if __name__ == '__main__':
    initialization()
    evaluate_on_validation_dataset()

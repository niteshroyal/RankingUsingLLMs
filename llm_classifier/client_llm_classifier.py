import os
import json
import torch
import logging
import requests

from conf import configuration
from llm_classifier.prepare_dataset import train_text, train_label


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def evaluate_on_validation_dataset():
    hits = []
    counter = 0
    batch_texts = []
    batch_answers = []
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            text = train_text(datapoint)
            answer = train_label(datapoint)
            batch_texts.append(text)
            batch_answers.append(answer)
            if len(batch_texts) == configuration.batch_size:
                data = {"prompts": batch_texts}
                resp = requests.post(configuration.server_url, json=data)
                resp = resp.json()
                eval_logits = resp["llm_response"]
                eval_logits = torch.tensor(eval_logits)

                # Taking maximum of the logits
                predictions = torch.argmax(eval_logits, dim=1)
                for i, prediction in enumerate(predictions):
                    p_answer = prediction.item()
                    answer = batch_answers[i]
                    if answer == p_answer:
                        hits.append(1)
                    else:
                        hits.append(0)
                counter += len(batch_texts)
                batch_texts = []
                batch_answers = []
                logging.info(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
                print(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
    if batch_texts:
        data = {"prompts": batch_texts}
        resp = requests.post(configuration.server_url, json=data)
        resp = resp.json()
        eval_logits = resp["llm_response"]
        eval_logits = torch.tensor(eval_logits)

        # Taking maximum of the logits
        predictions = torch.argmax(eval_logits, dim=1)
        for i, prediction in enumerate(predictions):
            p_answer = prediction.item()
            answer = batch_answers[i]
            if answer == p_answer:
                hits.append(1)
            else:
                hits.append(0)
        counter += len(batch_texts)
        logging.info(f"Final batch processed. Total processed {counter} datapoints.")
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}%"
    logging.info(text)
    print(text)


if __name__ == '__main__':
    initialization()
    evaluate_on_validation_dataset()

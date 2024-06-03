import os
import json
import random
import logging

from conf import configuration
from utils.utils import read_data


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_ranking_qa(label, record1, record2, label2):
    if label == 'mountainLabel':
        element1 = record1['mountainLabel']
        element2 = record2['mountainLabel']
        score1 = float(record1['elevation'])
        score2 = float(record2['elevation'])
        question = f'The next question is about two mountains. Does {element1} have higher elevation than {element2}?'
        if score1 > score2:
            answer = 'Yes'
        else:
            answer = 'No'
    # elif label == 'museumLabel':
    #     question = f'Is {element1} located at higher latitude compared to {element2}?'
    elif label == 'riverLabel':
        element1 = record1['riverLabel']
        element2 = record2['riverLabel']
        score1 = float(record1['length'])
        score2 = float(record2['length'])
        question = f'The following question is about two rivers. Answer the following with Yes or No. Is {element1} longer than {element2}?'
        if score1 > score2:
            answer = 'Yes'
        else:
            answer = 'No'
    elif label == 'elementSymbol':
        element1 = record1['elementLabel']
        element2 = record2['elementLabel']
        score1 = float(record1['minDiscovery'])
        score2 = float(record2['minDiscovery'])
        question = f'The following question is about two chemical elements: Should we rank {element1} older than {element2} in terms of date of discovery?'
        if score1 < score2:
            answer = 'Yes'
        else:
            answer = 'No'

    # elif label == 'cityLabel':
    #     question = f'Does {element1} have larger population than {element2}?'
    # elif label == 'personLabel' and label2 == 'personDescription':
    #     question = f'Was {element1} born before {element2}?'
    # elif label == 'itemLabel':
    #     question = f'Is {element1} taller than {element2}?'
    # elif label == 'islandLabel':
    #     question = f'Is {element1} larger than {element2} in area?'
    # elif label == 'companyLabel':
    #     question = f'Was {element1} founded before {element2}?'
    # elif label == 'personLabel' and label2 == 'maxSocialMediaFollower':
    #     question = f'Does {element1} have more social media follower than {element2}?'
    # elif label == 'speciesLabel':
    #     question = f'Is {element1} generally more heavier than {element2}?'
    else:
        raise Exception('Dataset label mismatch.')
    datapoint = dict()
    datapoint['question'] = question
    datapoint['answer'] = answer
    return datapoint


class Controller:

    def __init__(self):
        self.training_file_handler = None
        self.validation_file_handler = None
        random.seed(42)

    def write_dataset_for_fine_tuning(self, headers, data):
        topn = configuration.topn
        topn_training = int(0.8 * topn)
        topn_validation = int(0.2 * topn)
        label = headers[3]
        label2 = headers[4]
        n = len(data)
        assert n >= topn
        selected_data = data[0:topn]
        random.shuffle(selected_data)
        training_selected_data = selected_data[0:topn_training]
        validation_selected_data = selected_data[topn_training:topn]
        count = 0
        while count < configuration.num_training_datapoints:
            indices = list(range(topn_training))
            idx = random.choice(indices)
            indices.remove(idx)
            idx2 = random.choice(indices)
            record1 = training_selected_data[idx]
            record2 = training_selected_data[idx2]
            datapoint = get_ranking_qa(label, record1, record2, label2)
            json.dump(datapoint, self.training_file_handler)
            self.training_file_handler.write('\n')
            count += 1
        count = 0
        while count < configuration.num_validation_datapoints:
            indices = list(range(topn_validation))
            idx = random.choice(indices)
            indices.remove(idx)
            idx2 = random.choice(indices)
            record1 = validation_selected_data[idx]
            record2 = validation_selected_data[idx2]
            datapoint = get_ranking_qa(label, record1, record2, label2)
            json.dump(datapoint, self.validation_file_handler)
            self.validation_file_handler.write('\n')
            count += 1

    def control(self):
        with open(configuration.training_dataset, 'w') as self.training_file_handler, open(configuration.validation_dataset, 'w') as self.validation_file_handler:
            all_files = os.listdir(configuration.ranking_dataset_folder)
            txt_files = [file for file in all_files if file.endswith('.txt')]
            for file_name in txt_files:

                file_name = 'matched_outputRiver_WikiPageRank.txt'

                # if configuration.entity_type == 'mountains':
                #     file_name = 'unique_matched_MountainHeightWikiPageRank.txt'
                # elif configuration.entity_type == 'rivers':
                #     file_name = 'matched_outputRiver_WikiPageRank.txt'
                # else:
                #     raise Exception()
                file_path = os.path.join(configuration.ranking_dataset_folder, file_name)
                headers, data = read_data(file_path)
                self.write_dataset_for_fine_tuning(headers, data)
                break


if __name__ == '__main__':
    initialization()
    obj = Controller()
    obj.control()

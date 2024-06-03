import json

property_type = 'height'

if property_type == 'height':
    physical_properties_pairwise_judgements = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet'
                                               '/PhysicalProperties/500_pairwiseHeight.txt')
    physical_properties_validation_file = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet'
                                           '/heights_validation.jsonl')
elif property_type == 'size':
    physical_properties_pairwise_judgements = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet'
                                               '/PhysicalProperties/500_pairwiseSize.txt')
    physical_properties_validation_file = ('/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet'
                                           '/sizes_validation.jsonl')
else:
    raise Exception('Incorrect type')


def create_validation_file():
    data = []

    with open(physical_properties_pairwise_judgements, 'r') as file:
        for line in file:
            clean_line = line.strip().replace('\t', '').replace('\n', '')
            if clean_line:
                json_object = json.loads(clean_line)
                data.append(json_object)

    with open(physical_properties_validation_file, 'w', encoding='utf-8') as file_handler:
        for datapoint in data:
            record = dict()
            element1 = datapoint['obj_a']
            element2 = datapoint['obj_b']
            if property_type == 'height':
                question = f'This question is about two objects: Is {element1} taller than {element2}?'
                prompt1 = f'Is {element1} among the tallest objects?'
                prompt2 = f'Is {element2} among the tallest objects?'
            elif property_type == 'size':
                question = f'This question is about two objects: Is {element1} larger than {element2}?'
                prompt1 = f'Is {element1} among the largest objects?'
                prompt2 = f'Is {element2} among the largest objects?'
            else:
                raise Exception('Incorrect type')
            label = datapoint['label']
            if label == 1:
                answer = 'Yes'
            elif label == 0:
                answer = 'No'
            else:
                raise Exception('Unexpected label')
            record['element1'] = element1
            record['element2'] = element2
            record['prompt1'] = prompt1
            record['prompt2'] = prompt2
            record['question'] = question
            record['answer'] = answer
            record['1_rank_higher_than_2'] = answer

            json.dump(record, file_handler)
            file_handler.write('\n')


if __name__ == '__main__':
    create_validation_file()

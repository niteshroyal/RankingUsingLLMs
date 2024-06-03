from datasets import load_dataset

from conf import configuration

train_dataset = load_dataset('json', data_files=configuration.training_dataset, split='train')
eval_dataset = load_dataset('json', data_files=configuration.validation_dataset, split='train')


def train_text(example):
    text = example['question']
    return text


def pointwise_eval_text(example):
    text1 = example['prompt1']
    text2 = example['prompt2']
    return [text1, text2]


def train_label(example):
    label = example['answer']
    if label == "Yes":
        return 1
    elif label == "No":
        return 0
    else:
        raise Exception('Incorrect label')


def pointwise_eval_label(example):
    label = example['1_rank_higher_than_2']
    if label == "Yes":
        return 1.0
    elif label == "No":
        return 0.0
    else:
        raise Exception('Incorrect label')


def train_rank(example):
    rank = int(example['rank'])
    return rank


def train_group(example):
    group = example['group']
    return group

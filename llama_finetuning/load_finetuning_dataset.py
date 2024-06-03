from datasets import load_dataset

from conf import configuration

train_dataset = load_dataset('json', data_files=configuration.training_dataset, split='train')
eval_dataset = load_dataset('json', data_files=configuration.validation_dataset, split='train')


def formatting_func_for_train(example):
    text = (f"### Question: {example['question']}\n\n"
            f"### Answer: {example['answer']}\n"
            f'### End')
    return text


def formatting_func_for_eval(example):
    text = f"### Question: {example['question']}\n\n"
    return text

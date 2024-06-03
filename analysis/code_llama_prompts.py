import re
from conf import configuration


def get_score(score1, score2):
    if score1 > score2:
        answer = 'Yes'
    else:
        answer = 'No'
    return answer


def get_score_inverse(score1, score2):
    if score1 < score2:
        answer = 'Yes'
    else:
        answer = 'No'
    return answer


def get_prompt_taste(record1, record2, filename):
    datapoints = []
    if filename == 'food_Taste.txt':
        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sweet_Mean'])
        score2 = float(record2['Sweet_Mean'])
        question = ('# We are comparing the sweetness of two food items\n'
                    f'return sweet("{element1}") > sweet("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Salty_Mean'])
        score2 = float(record2['Salty_Mean'])
        question = ('# We are comparing the saltiness of two food items\n'
                    f'return salty("{element1}") > salty("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sour_Mean'])
        score2 = float(record2['Sour_Mean'])
        question = ('# We are comparing the sourness of two food items\n'
                    f'return sour("{element1}") > sour("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Bitter_Mean'])
        score2 = float(record2['Bitter_Mean'])
        question = ('# We are comparing the bitterness of two food items\n'
                    f'return bitter("{element1}") > bitter("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Umami_Mean'])
        score2 = float(record2['Umami_Mean'])
        question = ('# We are comparing the umaminess of two food items\n'
                    f'return umami("{element1}") > umami("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Fat_Mean'])
        score2 = float(record2['Fat_Mean'])
        question = ('# We are comparing the fattiness of two food items\n'
                    f'return fatty("{element1}") > fatty("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints


def get_prompt_movies(record1, record2, filename):
    datapoint = dict()
    pattern = r"NEW(.+?)_movie_titles1\.txt"
    if re.search(pattern, filename):
        keyword = re.search(pattern, filename).group(1).lower()
        element1 = record1['title']
        element2 = record2['title']
        score1 = float(record1['score'])
        score2 = float(record2['score'])
        question = (f'# We are comparing whether one movie is more {keyword} than the other\n'
                    f'return {keyword}("{element1}") > {keyword}("{element2}")')
        answer = get_score(score1, score2)
    else:
        raise Exception('Filename mismatch.')
    datapoint['question'] = question
    datapoint['answer'] = answer
    return [datapoint]


def get_prompt_books(record1, record2, filename):
    datapoint = dict()
    pattern = r"NEW_(.+?)_Book_titles1\.txt"
    if re.search(pattern, filename):
        keyword = re.search(pattern, filename).group(1).lower()
        element1 = record1['title']
        element2 = record2['title']
        score1 = float(record1['score'])
        score2 = float(record2['score'])
        question = (f'# We are comparing whether one book is more {keyword} than the other\n'
                    f'return {keyword}("{element1}") > {keyword}("{element2}")')
        answer = get_score(score1, score2)
    else:
        raise Exception('Filename mismatch.')
    datapoint['question'] = question
    datapoint['answer'] = answer
    return [datapoint]


def get_prompt_rocks(record1, record2, filename):
    datapoints = []
    if filename == 'rock_data.txt':
        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['lightness'])
        score2 = float(record2['lightness'])
        question = ('# We are comparing the lightness in color of two types of rocks.\n'
                    f'return light_in_color("{element1}") > light_in_color("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['grainSize'])
        score2 = float(record2['grainSize'])
        question = ('# We are comparing the coarseness of two types of rocks.\n'
                    f'return coarse("{element1}") > coarse("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['roughness'])
        score2 = float(record2['roughness'])
        question = ('# We are comparing the roughness of two types of rocks.\n'
                    f'return rough("{element1}") > rough("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['shine'])
        score2 = float(record2['shine'])
        question = ('# We are comparing the shininess of two types of rocks.\n'
                    f'return shiny("{element1}") > shiny("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['organization'])
        score2 = float(record2['organization'])
        question = ('# We are comparing the organization of two types of rocks.\n'
                    f'return organized("{element1}") > organized("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['variability'])
        score2 = float(record2['variability'])
        question = ('# We are comparing the variability in colour of two types of rocks.\n'
                    f'return variable_in_colour("{element1}") > variable_in_colour("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['density'])
        score2 = float(record2['density'])
        question = ('# We are comparing the density of two types of rocks.\n'
                    f'return density("{element1}") > density("{element2}")')
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints
import re


def get_score(score1, score2):
    if score1 > score2:
        answer = 'Yes'
    else:
        answer = 'No'
    return answer


def get_prompt_physical_properties(record1, record2, filename):
    datapoints = []
    if filename == '49_objectMass.txt':
        element1 = record1['item']
        element2 = record2['item']
        score1 = float(record1['mass'])
        score2 = float(record2['mass'])
        question = f'This question is about two objects: Is {element1} heavier than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the heaviest objects?'
        prompt2 = f'Is {element2} among the heaviest objects?'
        group = 'heavy objects'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Dataset label mismatch.')
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
        question = f'This question is about two movies: Is {element1} more {keyword} than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the most {keyword} movies?'
        prompt2 = f'Is {element2} among the most {keyword} movies?'
        group = f'{keyword} movies'
    else:
        raise Exception('Filename mismatch.')
    datapoint['prompt1'] = prompt1
    datapoint['prompt2'] = prompt2
    datapoint['score1'] = score1
    datapoint['score2'] = score2
    datapoint['answer'] = answer
    datapoint['group'] = group
    datapoint['question'] = question
    datapoint['element1'] = element1
    datapoint['element2'] = element2
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
        question = f'This question is about two books: Is {element1} more {keyword} than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the most {keyword} books?'
        prompt2 = f'Is {element2} among the most {keyword} books?'
        group = f'{keyword} books'
    else:
        raise Exception('Filename mismatch.')
    datapoint['prompt1'] = prompt1
    datapoint['prompt2'] = prompt2
    datapoint['score1'] = score1
    datapoint['score2'] = score2
    datapoint['answer'] = answer
    datapoint['group'] = group
    datapoint['question'] = question
    datapoint['element1'] = element1
    datapoint['element2'] = element2
    return [datapoint]


def get_prompt_taste(record1, record2, filename):
    datapoints = []
    if filename == 'food_Taste.txt':
        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sweet_Mean'])
        score2 = float(record2['Sweet_Mean'])
        question = f'This question is about two food items: Is {element1} generally sweeter in taste than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the sweetest food items?'
        prompt2 = f'Is {element2} among the sweetest food items?'
        group = 'sweetest food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Salty_Mean'])
        score2 = float(record2['Salty_Mean'])
        question = f'This question is about two food items: Is {element1} generally saltier than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the saltiest food items?'
        prompt2 = f'Is {element2} among the saltiest food items?'
        group = 'saltiest food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sour_Mean'])
        score2 = float(record2['Sour_Mean'])
        question = f'This question is about two food items: Is {element1} generally more sour in taste than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the sourest food items?'
        prompt2 = f'Is {element2} among the sourest food items?'
        group = 'sourest food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Bitter_Mean'])
        score2 = float(record2['Bitter_Mean'])
        question = f'This question is about two food items: Is {element1} generally more bitter in taste than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the bitterest food items?'
        prompt2 = f'Is {element2} among the bitterest food items?'
        group = 'bitterest food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Umami_Mean'])
        score2 = float(record2['Umami_Mean'])
        question = f'This question is about two food items: Is {element1} generally more umami than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the most umami food items?'
        prompt2 = f'Is {element2} among the most umami food items?'
        group = 'umami food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Fat_Mean'])
        score2 = float(record2['Fat_Mean'])
        question = f'This question is about two food items: Does {element1} taste fattier than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the fattiest food items?'
        prompt2 = f'Is {element2} among the fattiest food items?'
        group = 'fattiest food items'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints


def get_prompt_rocks(record1, record2, filename):
    datapoints = []
    if filename == 'rock_data.txt':
        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['lightness'])
        score2 = float(record2['lightness'])
        question = f'This question is about two types of rocks: Is {element1} lighter in color than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the lightest-colored rocks?'
        prompt2 = f'Is {element2} among the lightest-colored rocks?'
        group = 'lightest-colored rocks'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['grainSize'])
        score2 = float(record2['grainSize'])
        question = f'This question is about two types of rocks: Is {element1} more coarse than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the coarsest rocks?'
        prompt2 = f'Is {element2} among the coarsest rocks?'
        group = 'coarsest rocks'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['roughness'])
        score2 = float(record2['roughness'])
        question = f'This question is about two types of rocks: Is {element1} rougher than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the roughest rocks?'
        prompt2 = f'Is {element2} among the roughest rocks?'
        group = 'roughest rocks'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['shine'])
        score2 = float(record2['shine'])
        question = f'This question is about two types of rocks: Is {element1} more shiny than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the shiniest rocks?'
        prompt2 = f'Is {element2} among the shiniest rocks?'
        group = 'shiniest rocks'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['organization'])
        score2 = float(record2['organization'])
        question = f'This question is about two types of rocks: Does {element1} have a more uniform grain structure than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the rocks with the most uniform grain structure?'
        prompt2 = f'Is {element2} among the rocks with the most uniform grain structure?'
        group = 'rocks with the most uniform grain structure'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['variability'])
        score2 = float(record2['variability'])
        question = f'This question is about two types of rocks: Does {element1} have more variability in color than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the rocks with the greatest variability in color?'
        prompt2 = f'Is {element2} among the rocks with the greatest variability in color?'
        group = 'rocks with the greatest variability in color'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['density'])
        score2 = float(record2['density'])
        question = f'This question is about two types of rocks: Is {element1} denser than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the densest rocks?'
        prompt2 = f'Is {element2} among the densest rocks?'
        group = 'densest rocks'
        datapoint = dict()
        datapoint['element1'] = element1
        datapoint['element2'] = element2
        datapoint['prompt1'] = prompt1
        datapoint['prompt2'] = prompt2
        datapoint['score1'] = score1
        datapoint['score2'] = score2
        datapoint['group'] = group
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints


def get_wikidata_prompts(record1, record2, filename):
    if filename == 'matched_outputRiver_WikiPageRank.txt':
        element1 = record1['riverLabel']
        element2 = record2['riverLabel']
        score1 = float(record1['length'])
        score2 = float(record2['length'])
        question = f'This question is about two rivers: Is {element1} longer than {element2}?'
        prompt1 = f'Is {element1} among the longest rivers?'
        prompt2 = f'Is {element2} among the longest rivers?'
        answer = get_score(score1, score2)
        group = 'longest rivers'
    elif filename == 'matched_query_cityRankedPopulation_WikiRank.txt':
        element1 = record1['cityLabel']
        element2 = record2['cityLabel']
        score1 = float(record1['maxPopulation'])
        score2 = float(record2['maxPopulation'])
        question = f'This question is about two cities: Does {element1} have a larger population than {element2}?'
        prompt1 = f'Is {element1} among the most populated cities?'
        prompt2 = f'Is {element2} among the most populated cities?'
        answer = get_score(score1, score2)
        group = 'populated cities'
    elif filename == 'matched_query_PersonBorn_London_WikiRank.txt':
        element1 = record1['personLabel']
        element2 = record2['personLabel']
        score1 = float(record1['dob'])
        score2 = float(record2['dob'])
        question = f'This question is about two persons: Was {element1} born after {element2}?'
        prompt1 = f'Is {element1} among the youngest popular individuals?'
        prompt2 = f'Is {element2} among the youngest popular individuals?'
        answer = get_score(score1, score2)
        group = 'youngest people'
    elif filename == 'matched_query_Rank_BuildingHeight_WikiRank.txt':
        element1 = record1['itemLabel']
        element2 = record2['itemLabel']
        score1 = float(record1['maxHeight'])
        score2 = float(record2['maxHeight'])
        question = f'This question is about two buildings: Is {element1} taller than {element2}?'
        prompt1 = f'Is {element1} among the tallest buildings?'
        prompt2 = f'Is {element2} among the tallest buildings?'
        answer = get_score(score1, score2)
        group = 'tallest buildings'
    elif filename == 'matched_query_Rank_Island_Area_WikiRank.txt':
        element1 = record1['islandLabel']
        element2 = record2['islandLabel']
        score1 = float(record1['No_islandArea'])
        score2 = float(record2['No_islandArea'])
        question = f'This question is about two islands: Is {element1} larger than {element2} in area?'
        prompt1 = f'Is {element1} among the largest islands?'
        prompt2 = f'Is {element2} among the largest islands?'
        answer = get_score(score1, score2)
        group = 'largest islands'
    elif filename == 'matched_RankMusueumsLattitude_Italy_WikiRank.txt':
        element1 = record1['museumLabel']
        element2 = record2['museumLabel']
        score1 = float(record1['Rank_lat'])
        score2 = float(record2['Rank_lat'])
        question = f'This question is about two museums in Italy: Is {element1} located at a higher latitude compared to {element2}?'
        prompt1 = f'Is {element1} among the museums with the highest latitude in Italy?'
        prompt2 = f'Is {element2} among the museums with the highest latitude in Italy?'
        answer = get_score(score1, score2)
        group = 'highest latitude'
    elif filename == 'unique_matched_InceptionCompanyWikiPageRank.txt':
        element1 = record1['companyLabel']
        element2 = record2['companyLabel']
        score1 = float(record1['minInception'])
        score2 = float(record2['minInception'])
        question = f'This question is about two companies: Was {element1} founded after {element2}?'
        prompt1 = f'Is {element1} among the most recently founded companies?'
        prompt2 = f'Is {element2} among the most recently founded companies?'
        answer = get_score(score1, score2)
        group = 'recently founded'
    elif filename == 'unique_matched_MountainHeightWikiPageRank.txt':
        element1 = record1['mountainLabel']
        element2 = record2['mountainLabel']
        score1 = float(record1['elevation'])
        score2 = float(record2['elevation'])
        question = f'This question is about two mountains: Does {element1} have a higher elevation than {element2}?'
        prompt1 = f'Is {element1} among the mountains with the highest elevation?'
        prompt2 = f'Is {element2} among the mountains with the highest elevation?'
        answer = get_score(score1, score2)
        group = 'highest elevation'
    elif filename == 'unique_matched_Person_SOcialMedia_WikiPageRank.txt':
        element1 = record1['personLabel']
        element2 = record2['personLabel']
        score1 = float(record1['maxSocialMediaFollower'])
        score2 = float(record2['maxSocialMediaFollower'])
        question = f'This question is about two persons: Does {element1} have more social media followers than {element2}?'
        prompt1 = f'Is {element1} among the people with the highest number of social media followers?'
        prompt2 = f'Is {element2} among the people with the highest number of social media followers?'
        answer = get_score(score1, score2)
        group = 'highest social media followers'
    elif filename == 'unique_matched_speciesMass_WikiPageRank.txt':
        element1 = record1['speciesLabel']
        element2 = record2['speciesLabel']
        score1 = float(record1['maxMass'])
        score2 = float(record2['maxMass'])
        question = f'This question is about two species: Is {element1} generally heavier than {element2}?'
        prompt1 = f'Is {element1} among the heaviest species?'
        prompt2 = f'Is {element2} among the heaviest species?'
        answer = get_score(score1, score2)
        group = 'heaviest species'
    elif filename == 'matched_queryAcademyAward_Direction_WikiRank.txt':
        element1 = record1['directorLabel']
        element2 = record2['directorLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two directors: Has {element1} won more Academy Awards than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the directors with the most Academy Awards?'
        prompt2 = f'Is {element2} among the directors with the most Academy Awards?'
        group = 'directors with the most Academy Awards'
    elif filename == 'matched_query_awardActor_WikiRank.txt':
        element1 = record1['actorLabel']
        element2 = record2['actorLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two actors: Has {element1} won more Academy Awards for Best Actor than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the actors who have won the most Academy Awards for Best Actor?'
        prompt2 = f'Is {element2} among the actors who have won the most Academy Awards for Best Actor?'
        group = 'actors with the most Academy Awards'
    elif filename == 'matched_query_chemcialELements_DIscovery_WikiPageRank.txt':
        element1 = record1['elementLabel']
        element2 = record2['elementLabel']
        score1 = float(record1['minDiscovery'])
        score2 = float(record2['minDiscovery'])
        question = f'This question is about two chemical elements: Was {element1} discovered after {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the recently discovered chemical elements?'
        prompt2 = f'Is {element2} among the recently discovered chemical elements?'
        group = 'recently discovered chemical elements'
    elif filename == 'matched_queryfood_WaterFootPrint_WikiRank.txt':
        element1 = record1['foodGrpLabel']
        element2 = record2['foodGrpLabel']
        score1 = float(record1['WaterFootPrint'])
        score2 = float(record2['WaterFootPrint'])
        question = f'This question is about two types of food: Does {element1} have a larger water footprint than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the foods with the largest water footprint?'
        prompt2 = f'Is {element2} among the foods with the largest water footprint?'
        group = 'largest water footprint'
    elif filename == 'matched_query_noGrammyAward_Composer_WikiRank.txt':
        element1 = record1['artistLabel']
        element2 = record2['artistLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two artists: Has {element1} received more awards than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the artists who have won the most awards?'
        prompt2 = f'Is {element2} among the artists who have won the most awards?'
        group = 'artists with most awards'
    elif filename == 'matched_query_rankCountries_Population.txt':
        element1 = record1['countryLabel']
        element2 = record2['countryLabel']
        score1 = float(record1['maxPopulation'])
        score2 = float(record2['maxPopulation'])
        question = f'This question is about two countries: Does {element1} have a larger population than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the most populous countries?'
        prompt2 = f'Is {element2} among the most populous countries?'
        group = 'most populous countries'
    elif filename == 'matched_query_rankElements_AtomicNo_WikiPageRank.txt':
        element1 = record1['elementLabel']
        element2 = record2['elementLabel']
        score1 = float(record1['atomicNo'])
        score2 = float(record2['atomicNo'])
        question = f'This question is about two chemical elements: Does {element1} have a higher atomic number than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the chemical elements with the highest atomic numbers?'
        prompt2 = f'Is {element2} among the chemical elements with the highest atomic numbers?'
        group = 'highest atomic numbers'
    elif filename == 'matched_query_schoville_WikiRank.txt':
        element1 = record1['foodName']
        element2 = record2['foodName']
        score1 = float(record1['Rank_scovilleGrade'])
        score2 = float(record2['Rank_scovilleGrade'])
        question = f'This question is about two types of food: Does {element1} have a higher Scoville grade than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the foods with the highest Scoville grade?'
        prompt2 = f'Is {element2} among the foods with the highest Scoville grade?'
        group = 'highest Scoville grade'
    elif filename == 'matched_RankBuildingElevators_WikiRank.txt':
        element1 = record1['buildingLabel']
        element2 = record2['buildingLabel']
        score1 = float(record1['no_elevator'])
        score2 = float(record2['no_elevator'])
        question = f'This question is about two buildings: Does {element1} have more elevators than {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among the buildings with the highest number of elevators?'
        prompt2 = f'Is {element2} among the buildings with the highest number of elevators?'
        group = 'highest number of elevators'
    elif filename == 'matched_rankMusicalObjects_ByInceptionDate.txt':
        element1 = record1['instrumentLabel']
        element2 = record2['instrumentLabel']
        score1 = float(record1['inceptDate'])
        score2 = float(record2['inceptDate'])
        question = f'This question is about two instruments: Did {element1} came into existence after {element2}?'
        answer = get_score(score1, score2)
        prompt1 = f'Is {element1} among of the instruments that have recently come into existence?'
        prompt2 = f'Is {element2} among of the instruments that have recently come into existence?'
        group = 'instruments recently come into existence'
    else:
        raise Exception('Dataset label mismatch.')
    datapoint = dict()
    datapoint['prompt1'] = prompt1
    datapoint['prompt2'] = prompt2
    datapoint['score1'] = score1
    datapoint['score2'] = score2
    datapoint['answer'] = answer
    datapoint['group'] = group
    datapoint['question'] = question
    datapoint['element1'] = element1
    datapoint['element2'] = element2
    return datapoint

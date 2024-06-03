import json

# Path to the dataset
dataset_path = "/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/llm_validation_taste_mixed_500_datapoints.jsonl"

# Keywords for each taste
taste_keywords = {
    'salty': 'saltier than',
    'umami': 'umami than',
    'fatty': 'fattier than',
    'bitter': 'bitter in taste',
    'sweet': 'sweeter in taste',
    'sour': 'sour in taste'
}

# Create or open files for each taste category in the specified folder
folder_path = "/scratch/c.scmnk4/elexir/ranking_research/resources/Ranking_DataSet/"
files = {taste: open(f"{folder_path}{taste}.jsonl", "a") for taste in taste_keywords}

with open(dataset_path, "r") as dataset:
    for line in dataset:
        datapoint = json.loads(line)
        question = datapoint['question'].lower()

        # Determine the taste mentioned in the question based on keywords
        for taste, keyword in taste_keywords.items():
            if keyword in question:
                # Write the line to the corresponding file
                files[taste].write(line)
                break  # Move to the next line after writing to a file

# Close all files
for file in files.values():
    file.close()

print("Data separation into taste categories completed.")

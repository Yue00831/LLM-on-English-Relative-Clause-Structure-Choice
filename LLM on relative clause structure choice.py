!pip3 install minicons
from minicons import scorer
gpt_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu') #GPT-2; cpu -> cuda [for GPU]

import pandas as pd 
import numpy as np 

# take the imput data: feed data
import pandas as pd 
file_path = '/your input excel sheet.xlsx'
data = pd.read_excel(file_path)

#### Goal: generate the entire prompt's surprisal score and store them accordingly
def calculate_surprisal_score(prompt):
    return 0.0

# Initialize columns for the output file
output_columns = [
    'item', 'pictures_ID', 
    'prompt_v1', 'prompt_v1_alternative', 'surprisal_sum_v1', 'surprisal_sum_v1_alternative',
    'prompt_v2', 'prompt_v2_alternative', 'surprisal_sum_v2', 'surprisal_sum_v2_alternative',
    'prompt_v3', 'prompt_v3_alternative', 'surprisal_sum_v3', 'surprisal_sum_v3_alternative',
    'prompt_v4', 'prompt_v4_alternative', 'surprisal_sum_v4', 'surprisal_sum_v4_alternative'
]
output_df = pd.DataFrame(columns=output_columns)

# loop to process each row in the input data
for idx, row in data.iterrows():
    output_row = {'item': row['item'], 'pictures_ID': row['pictures_ID']}  
    for v in range(1, 5):  # there are 4 versions of prompts and each has an alternative
        prompt_key = f'prompt_v{v}'
        alternative_key = f'prompt_v{v}_alternative'
        output_row[prompt_key] = f"{row['description']} {row['question']} {row[f'prediction_v{v}']}"
        output_row[alternative_key] = f"{row['description']} {row['question']} {row[f'prediction_v{v}_alternative']}"
        
        # get sum surprisal scores
        output_row[f'surprisal_sum_v{v}'] = gpt_model.sequence_score(output_row[prompt_key], reduction = lambda x: -x.sum(0).item())
        output_row[f'surprisal_sum_v{v}_alternative'] = gpt_model.sequence_score(output_row[alternative_key], reduction = lambda x: -x.sum(0).item())
    
    # Append the processed row to the output DataFrame
    output_df = output_df.append(output_row, ignore_index=True)

# Save the outputto an Excel file
output_path = 'your desired output location.xlsx'  # Adjust this to where you want to save the output
output_df.to_excel(output_path, index=False)

print("Surprisal_sum Generated!")



# take the output of the above code, we are going to calculate each word's surprisal score because sometimes the sum does not tell us much since each sentence is in different length.
# Load the Excel file containing the prompts
data_path = 'your output excel sheet from the last step.xlsx' 
data = pd.read_excel(data_path)

# Goal: generate every word's surprisal score for every prompt. 
output_columns = [
    'item', 'pictures_ID',
    'prompt_v1', 'prompt_v1_alternative', 'suprisal_token_v1', 'suprisal_token_v1_alternative',
    'prompt_v2', 'prompt_v2_alternative', 'suprisal_token_v2', 'suprisal_token_v2_alternative',
    'prompt_v3', 'prompt_v3_alternative', 'suprisal_token_v3', 'suprisal_token_v3_alternative',
    'prompt_v4', 'prompt_v4_alternative', 'suprisal_token_v4', 'suprisal_token_v4_alternative',
]
output_df = pd.DataFrame(columns=output_columns)

def calculate_token_scores(prompt):
    scores = gpt_model.token_score([prompt], surprisal=True, base_two=True)
    return scores

for idx, row in data.iterrows():
    output_row = {'item': row['item'], 'pictures_ID': row['pictures_ID']} 
    for v in range(1, 5):  # For each of the 4 versions and their alternatives
        for alt in ['', '_alternative']:
            prompt_key = f'prompt_v{v}{alt}'
            scores_key = f'suprisal_token_v{v}{alt}'
            prompt = row[prompt_key]
            # assign the prompt text to the output row, ensuring prompts are next to their scores
            output_row[prompt_key] = prompt  # This ensures the prompt column is populated
            
            # Check if the prompt is not NaN before calculating scores
            if pd.notna(prompt):
                # Calculate individual word's score
                scores = calculate_token_scores(prompt)
                output_row[scores_key] = scores  # Assign calculated scores next to the prompt

    # Append the processed row to the output DataFrame
    output_df = output_df.append(output_row, ignore_index=True)

# Save the output file
output_path = 'your desired output file.xlsx'
output_df.to_excel(output_path, index=False)

print("Surprisal_Token Generated!")


#### To calculate mean surpr score
output_columns = [
    'item', 'pictures_ID',
    'prompt_v1', 'prompt_v1_alternative', 'suprisal_mean_v1', 'suprisal_mean_v1_alternative',
    'prompt_v2', 'prompt_v2_alternative', 'suprisal_mean_v2', 'suprisal_mean_v2_alternative',
    'prompt_v3', 'prompt_v3_alternative', 'suprisal_mean_v3', 'suprisal_mean_v3_alternative',
    'prompt_v4', 'prompt_v4_alternative', 'suprisal_mean_v4', 'suprisal_mean_v4_alternative',
]
output_df = pd.DataFrame(columns=output_columns)


def calculate_mean_scores(prompt):
    scores = gpt_model.sequence_score([prompt], reduction = lambda x: -x.mean(0).item())
    return round(scores[0], 2)

for idx, row in data.iterrows():
    output_row = {'item': row['item'], 'pictures_ID': row['pictures_ID']} 
    for v in range(1, 5):  # For each of the 4 versions and their alternatives
        for alt in ['', '_alternative']:
            prompt_key = f'prompt_v{v}{alt}'
            scores_key = f'suprisal_mean_v{v}{alt}'
            prompt = row[prompt_key]
            # assign the prompt text to the output row, ensuring prompts are next to their scores
            output_row[prompt_key] = prompt  # This ensures the prompt column is populated
            
            # Check if the prompt is not NaN before calculating scores
            if pd.notna(prompt):
                # Calculate individual word's score
                scores = calculate_mean_scores(prompt)
                output_row[scores_key] = scores  # Assign calculated scores next to the prompt

    # Append the processed row to the output DataFrame
    output_df = output_df.append(output_row, ignore_index=True)

# Save the output file
output_path = 'your desired output file.xlsx'
output_df.to_excel(output_path, index=False)

print("Surprisal_Mean Generated!")


#### Using bidirectional BERT to generate surprisal score

from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd
import numpy as np

# Load any Excel file containing the prompts
data_path = 'your input file.xlsx' 
data = pd.read_excel(data_path)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to('cpu')  # using CPU


### testing BERT using individual sentences
# Define function to calculate mean surprisal scores using BERT for a list of prompts
def calculate_mean_scores_bert(prompts):
    scores = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt") #tells the tokenizer to return PyTorch tensors.
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        scores.append(round(loss.item(), 2))
    return scores

stimuli_1hold = ["The keys to the cells unsurprisingly were rusty.",
           "The key to the cells unsurprisingly were rusty."]

print(calculate_mean_scores_bert(stimuli_1hold))



#### If the output result looks fine, run the code for your excel input
def calculate_mean_scores_bert(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return round(loss.item(), 2)

output_columns = [
    'item', 'pictures_ID',
    'prompt_v1', 'prompt_v1_alternative', 'suprisal_mean_v1', 'suprisal_mean_v1_alternative',
    'prompt_v2', 'prompt_v2_alternative', 'suprisal_mean_v2', 'suprisal_mean_v2_alternative',
    'prompt_v3', 'prompt_v3_alternative', 'suprisal_mean_v3', 'suprisal_mean_v3_alternative',
    'prompt_v4', 'prompt_v4_alternative', 'suprisal_mean_v4', 'suprisal_mean_v4_alternative',
]
output_df = pd.DataFrame(columns=output_columns)

for idx, row in data.iterrows():
    output_row = {'item': row['item'], 'pictures_ID': row['pictures_ID']}
    for v in range(1, 5):  # For each of the 4 versions and their alternatives
        for alt in ['', '_alternative']:
            prompt_key = f'prompt_v{v}{alt}'
            scores_key = f'suprisal_mean_v{v}{alt}'
            prompt = row[prompt_key]
            output_row[prompt_key] = prompt
            
            # Check if the prompt is not NaN before calculating scores
            if pd.notna(prompt):
                scores = calculate_mean_scores_bert(prompt)
                output_row[scores_key] = scores

    # Append the processed row to the output DataFrame
    output_df = output_df.append(output_row, ignore_index=True)

output_path = 'suprisal_mean_BERT.xlsx'
output_df.to_excel(output_path, index=False)
print("Surprisal_Mean Generated using BERT!")
import json
import sys
import requests
from tqdm import tqdm
import os

#run this program in terminal: python3 classify.py dataset.jsonl API_URL [TOKEN]

def api_call(text, api_url, token=None):
    payload = {'text': text}
    if token:
        payload['token'] = token
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        data = response.json()
        if data and 'label' in data:
            return data['label']
    else:
        print("Error "+ str(response.status_code))

def main(dataset_path, api_url, token=None):
    with open(dataset_path, 'r') as file:
        dataset = file.readlines()

    modified_dataset = []  

    with tqdm(total=len(dataset), desc="Processing") as pbar:
        for line in dataset:
            data = json.loads(line)
            text = data['text']
            api_label = api_call(text, api_url, token)
            data['classification_output'] = api_label  

            modified_dataset.append(json.dumps(data))
            pbar.update(1)  

    with open('modified_dataset.jsonl', 'w') as output_file:
        output_file.write('\n'.join(modified_dataset))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 classify.py <dataset_path> <API_URL> [TOKEN]")
    else:
        dataset_path = sys.argv[1]
        api_url = sys.argv[2]
        token = sys.argv[3] if len(sys.argv) > 3 else None
        main(dataset_path, api_url, token)


import json
import sys
import requests

#run this program in terminal: python3 classify.py dataset.jsonl

def api_call(text):
    url = 'http://10.19.3.26:8000/classify'
    payload = {'text': text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        if data and 'label' in data[0]:
            return data[0]['label']
    else:
        print("Error "+ str(response.status_code))

def main(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = file.readlines()

    results = []
    total_entries = len(dataset)
    processed_entries = 0
    for line in dataset:
        item = json.loads(line)
        text = item['text']
        label = api_call(text)
        item['classification_output'] = label
        results.append(item)

        processed_entries += 1
        progress = (processed_entries / total_entries) * 100
        print(f"Progress: {progress:.2f}%")

    output_path = 'output.jsonl'
    with open(output_path, 'w') as file:
        file.write('\n'.join(json.dumps(item) for item in results))

    print(f"Classification labels added to the dataset. Saved at: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 classify.py <dataset_path>")
    else:
        dataset_path = sys.argv[1]
        main(dataset_path)

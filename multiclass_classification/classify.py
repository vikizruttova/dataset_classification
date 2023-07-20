import json
import sys
import requests

#run this program in terminal: python3 classify.py dataset.jsonl API_endpoint [TOKEN]

def api_call(text, api_url, token=None):
    payload = {'text': text}
    if token:
        payload['token'] = token
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        data = response.json()
        if data and 'label' in data[0]:
            return data[0]['label']
    else:
        print("Error "+ str(response.status_code))

def main(dataset_path, api_url, token=None):
    with open(dataset_path, 'r') as file:
        dataset = file.readlines()

    results = []
    total_entries = len(dataset)
    processed_entries = 0
    for line in dataset:
        item = json.loads(line)
        text = item['text']
        label = api_call(text, api_url, token)
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
    if len(sys.argv) < 3:
        print("Usage: python3 classify.py <dataset_path> <API_endpoint> [TOKEN]")
    else:
        dataset_path = sys.argv[1]
        api_url = sys.argv[2]
        token = sys.argv[3] if len(sys.argv) > 3 else None
        main(dataset_path, api_url, token)


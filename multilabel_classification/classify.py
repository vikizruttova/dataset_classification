import json
import sys
import requests

def api_call(text, url):
    payload = {'text': text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        filtered_data = [label for label in data if label['score'] > 0.29]
        return filtered_data
    else:
        print("Error " + str(response.status_code))

def main(dataset_path, output_path, url):
    with open(dataset_path, 'r') as file:
        dataset = file.readlines()

    with open(output_path, 'w') as output_file:
        for line in dataset:
            item = json.loads(line)
            text = item['text']
            label = api_call(text, url)
            item['classification_output'] = label
            output_file.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 classify_multilabel.py <dataset_path> <output_path> <api_url>")
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
        api_url = sys.argv[3]
        main(dataset_path, output_path, api_url)


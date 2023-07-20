import csv
import json
import sys
#this program transforms the dataset's scores into binary values and returns both jsonl and csv files
def extract_labels(input_file):
    all_labels = set()
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            labels = entry.get("labels", [])
            for l in labels:
                all_labels.add(l["label"])

            classification_output = entry.get("classification_output", [])
            for l in classification_output:
                all_labels.add(l["label"])

    return list(all_labels)

def process_dataset(input_file, output_file, LABELS):
    output = []
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            labels = entry.get("labels", [])
            binary_labels = {}
            for label in LABELS:
                if any(l["label"] == label for l in labels):
                    score = next((l["score"] for l in labels if l["label"] == label), 0)
                    binary_labels[label] = 1 if score != 0 else 0
                else:
                    binary_labels[label] = 0

            classification_output = entry.get("classification_output", [])
            binary_classification_output = {}
            for label in LABELS:
                if any(l["label"] == label for l in classification_output):
                    score = next((l["score"] for l in classification_output if l["label"] == label), 0)
                    binary_classification_output[label] = 1 if score != 0 else 0
                else:
                    binary_classification_output[label] = 0

            entry["labels"] = binary_labels
            entry["classification_output"] = binary_classification_output

            output.append(entry)


    with open(output_file + '.jsonl', 'w') as file:
        for entry in output:
            file.write(json.dumps(entry))
            file.write('\n')

    with open(output_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["text"] + LABELS)
        for entry in output:
            text = entry["text"]  
            row = [text] + [entry["labels"].get(label, 0) for label in LABELS]
            writer.writerow(row)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 transformer.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    LABELS = extract_labels(input_file)

    process_dataset(input_file, output_file, LABELS)

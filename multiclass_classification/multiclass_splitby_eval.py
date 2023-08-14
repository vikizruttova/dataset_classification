import sys
import os
import json
import jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


def read_data(file_path):
    y_true, y_pred = [], []

    with open(file_path, 'r') as file:
        for line in tqdm(file, desc="Reading file"):
            data = json.loads(line)
            true_label = data.get("labels")
            predicted_label = data.get('classification_output')

            if true_label is not None and predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(predicted_label)

    return y_true, y_pred


def generate_classification_report(y_true, y_pred, output_path):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Generating the classification report
    report = classification_report(y_true, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
    report_file = os.path.join(output_path, 'classification_report.jsonl')
    with jsonlines.open(report_file, 'w') as writer:
        for label in unique_labels:
            label_report = {label: report[label]}
            writer.write(label_report)

    # Plotting the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=unique_labels)
    conf_df = pd.DataFrame(conf_mat, index=unique_labels, columns=unique_labels)

    plt.figure(figsize=(20, 12))
    sns.heatmap(conf_df, annot=True, fmt='d', annot_kws={'fontsize': 10}, cmap='summer')
    sns.set(font_scale=0.7)
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.ylabel('True Labels', fontsize=10)
    plt.subplots_adjust(left=0.3, bottom=0.3)
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))

    print(f"Classification report saved at: {report_file}")
    print(f"Confusion matrix plot saved at: {os.path.join(output_path, 'confusion_matrix.png')}")


def get_unique_values_for_field(file_path, field):
    """Get all unique values for a specified field in the dataset."""
    unique_values = set()

    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=f"Extracting unique values for {field}"):
            data = json.loads(line)
            value = data.get(field)
            if value:
                unique_values.add(value)

    return list(unique_values)


def filter_data_by_field(file_path, field, value):
    """Filter data by a specific field-value pair."""
    y_true, y_pred = [], []

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data.get(field) == value:
                true_label = data.get('labels')
                predicted_label = data.get('classification_output')
                if true_label is not None and predicted_label is not None:
                    y_true.append(true_label)
                    y_pred.append(predicted_label)

    return y_true, y_pred


def main(file_path, output_path, split_by_field='lang'):
    os.makedirs(output_path, exist_ok=True)

    unique_values = get_unique_values_for_field(file_path, split_by_field)

    for value in unique_values:
        print(f"Processing for {split_by_field} = {value}")
        y_true, y_pred = filter_data_by_field(file_path, split_by_field, value)
        value_output_path = os.path.join(output_path, value)
        os.makedirs(value_output_path, exist_ok=True)
        generate_classification_report(y_true, y_pred, value_output_path)

        print(f"Reports for {split_by_field} = {value} saved in {value_output_path}")
        print("-" * 50)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 multiclass_split_eval.py <dataset_path> <output_path> [<split_by_field>]")
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
        split_by_field = sys.argv[3] if len(sys.argv) > 3 else 'lang'
        main(dataset_path, output_path, split_by_field)






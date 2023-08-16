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

def evaluate(conf_mat, labels, output_path, name):
    evaluation = {}
    for i, true_label in enumerate(labels):
        class_size = int(np.sum(conf_mat[i]))
        confusion = {}
        for j, predicted_label in enumerate(labels):
            if conf_mat[i, j] > 0 and i != j:
                confusion[predicted_label] = int(conf_mat[i, j])
        evaluation[true_label] = {'class size': class_size, 'confusion': confusion}

    evaluation_file = os.path.join(output_path, name)
    with open(evaluation_file, 'w') as file:
        json.dump(evaluation, file, indent=4)
    print("Evaluation results saved at:", evaluation_file)

    return evaluation

def row_mistakes(y_true, y_pred, output_path, name):
    rows = {}
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        if true_label != pred_label:
            if true_label not in rows:
                rows[true_label] = {'class size': 0, 'confusion': {}}
            if pred_label not in rows[true_label]['confusion']:
                rows[true_label]['confusion'][pred_label] = []
            rows[true_label]['confusion'][pred_label].append(i+1)
            rows[true_label]['class size'] += 1

    row_file = os.path.join(output_path, name)
    with open(row_file, 'w') as file:
        json.dump(rows, file, indent=4)
    print("Row mistakes results saved at:", row_file)


def generate_classification_report(y_true, y_pred, output_path, ax=None):
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

    if ax is None:
        plt.figure(figsize=(20, 12))
        ax = plt.gca()

    sns.heatmap(conf_df, annot=True, fmt='d', annot_kws={'fontsize': 10}, cmap='summer', ax=ax)
    sns.set(font_scale=0.7)
    ax.set_xlabel('Predicted Labels', fontsize=10)
    ax.set_ylabel('True Labels', fontsize=10)


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

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(unique_values))))
    subplot_height = 8  # Adjust as needed
    total_height = subplot_height * grid_size
    total_width = subplot_height * grid_size
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(total_width, total_height))

    for idx, value in enumerate(unique_values):
        print(f"Processing for {split_by_field} = {value}")
        
        # Filter the data
        y_true, y_pred = filter_data_by_field(file_path, split_by_field, value)
        
        # Create a unique directory for this split
        value_output_path = os.path.join(output_path, value)
        os.makedirs(value_output_path, exist_ok=True)
        
        # Get ax for current plot
        ax = axs[idx // grid_size, idx % grid_size]
        ax.set_title(value)  # Set title for the current split
        
        # Generate the classification report and plot the confusion matrix for this split
        generate_classification_report(y_true, y_pred, value_output_path, ax=ax)
        
        # Evaluate and note down mistakes for this split
        result_matrix = confusion_matrix(y_true, y_pred)
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        evaluate(result_matrix, unique_labels, value_output_path, "evaluation.json")
        row_mistakes(y_true, y_pred, value_output_path, 'mistakes.json')
        
        print(f"Reports for {split_by_field} = {value} saved in {value_output_path}")
        print("-" * 50)

    # Hide any unused subplots
    for i in range(len(unique_values), grid_size*grid_size):
        axs[i // grid_size, i % grid_size].axis('off')

    # After processing all splits, save the combined figure with all confusion matrices
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'all_matrices.png'))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 multiclass_split_eval.py <dataset_path> <output_path> [<split_by_field>]")
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
        split_by_field = sys.argv[3] if len(sys.argv) > 3 else 'lang'
        main(dataset_path, output_path, split_by_field)
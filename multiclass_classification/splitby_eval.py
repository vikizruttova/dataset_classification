import sys
import io
import os
import json
import jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from clarity_matrix import generate_clarity_matrix
import itertools

def plot_figure_on_ax(fig, ax):
    """
    Plot the contents of one figure onto a given axis.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = plt.imread(buf)
    ax.imshow(img_arr)
    ax.axis('off') 

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
    """
    finds the id (the number of the row) where the mistake is
    """
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


def generate_classification_report(y_true, y_pred, output_path):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
    report_file = os.path.join(output_path, 'classification_report.jsonl')
    with jsonlines.open(report_file, 'w') as writer:
        for label in unique_labels:
            label_report = {label: report[label]}
            writer.write(label_report)
    return unique_labels


def get_unique_values_for_fields(file_path, fields):
    """Get all unique values for specified fields in the dataset."""
    unique_values = {field: set() for field in fields}

    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=f"Extracting unique values for {', '.join(fields)}"):
            data = json.loads(line)
            for field in fields:
                value = data.get(field)
                if value:
                    unique_values[field].add(value)

    return {field: list(values) for field, values in unique_values.items()}


def filter_data_by_fields(file_path, field_values):
    """Filter data by specific field-value pairs."""
    y_true, y_pred = [], []

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if all(data.get(field) == value for field, value in field_values.items()):
                true_label = data.get('labels')
                predicted_label = data.get('classification_output')
                if true_label is not None and predicted_label is not None:
                    y_true.append(true_label)
                    y_pred.append(predicted_label)

    return y_true, y_pred


def main(file_path, output_path, split_by_fields=['lang']):
    os.makedirs(output_path, exist_ok=True)

    if not split_by_fields:
        y_true, y_pred = read_data(file_path)
        value_output_path = os.path.join(output_path, "whole_dataset")
        os.makedirs(value_output_path, exist_ok=True)
        unique_labels = generate_classification_report(y_true, y_pred, value_output_path)
        result_matrix = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred))))
        evaluate(result_matrix, unique_labels, value_output_path, "evaluation.json")
        row_mistakes(y_true, y_pred, value_output_path, 'mistakes.json')
        
        clarity_fig = generate_clarity_matrix(y_true, y_pred, unique_labels)
        clarity_save_path = os.path.join(value_output_path, 'clarity_matrix.png')
        clarity_fig.savefig(clarity_save_path)
        plt.close(clarity_fig)
        return

    unique_values_dict = get_unique_values_for_fields(file_path, split_by_fields)

    all_combinations = [dict(zip(unique_values_dict, combo)) for combo in itertools.product(*unique_values_dict.values())]

    # get the grid size
    grid_size = int(np.ceil(np.sqrt(len(all_combinations))))

    subplot_height = 8  # Adjust as needed
    total_height = subplot_height * grid_size
    total_width = subplot_height * grid_size
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(total_width, total_height))

    for idx, combo in enumerate(all_combinations):
        combo_name = "_".join([f"{k}={v}" for k, v in combo.items()])
        print(f"Processing for {combo_name}")

        y_true, y_pred = filter_data_by_fields(file_path, combo)
        #create directory for the split
        value_output_path = os.path.join(output_path, combo_name)
        os.makedirs(value_output_path, exist_ok=True)

        unique_labels = generate_classification_report(y_true, y_pred, value_output_path)
        result_matrix = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred))))
        evaluate(result_matrix, unique_labels, value_output_path, "evaluation.json")
        row_mistakes(y_true, y_pred, value_output_path, 'mistakes.json')
        
        # get ax for current plot
        ax = axs[idx // grid_size, idx % grid_size]
        ax.set_title(combo_name)  # title
        clarity_fig = generate_clarity_matrix(y_true, y_pred, unique_labels)
        clarity_save_path = os.path.join(value_output_path, 'clarity_matrix.png')
        clarity_fig.savefig(clarity_save_path)
        plot_figure_on_ax(clarity_fig, ax)
        plt.close(clarity_fig)

    
        print(f"Reports for combination = {combo_name} saved in {value_output_path}")
        print("-" * 50)

    # hide unused subplots
    for i in range(len(all_combinations), grid_size*grid_size):
        axs[i // grid_size, i % grid_size].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'all_matrices.png'))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 multiclass_split_eval.py <dataset_path> <output_path> [<split_by_field1> <split_by_field2> ...]")
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
        split_by_fields = sys.argv[3:] if len(sys.argv) > 3 else None
        main(dataset_path, output_path, split_by_fields)
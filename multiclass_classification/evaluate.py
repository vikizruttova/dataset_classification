import sys
import os
import json
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def process_dataset(file_path):
    y_true = []
    y_pred = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['text']
            label = data['labels'][0]['label'] #  change this based on the dataset hierarchy
            api_label = data['classification_output']
            y_true.append(label)
            y_pred.append(api_label)
            if label not in labels:
                labels.append(label)
            if api_label not in labels:
                labels.append(api_label)

    return y_true, y_pred, labels

def process_dataset_general(file_path):
    y_true = []
    y_pred = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['text']
            label = data['labels'][0]['label']
            api_label = data['classification_output']
            label1 = label.split('.')[0]
            api_label1 = api_label.split('.')[0]

            if label1 not in labels:
                labels.append(label1)
            if api_label1 not in labels:
                labels.append(api_label1)

            y_true.append(label1)
            y_pred.append(api_label1)

    return y_true, y_pred, labels


def matrix(y_true, y_pred, labels):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    report_file = os.path.join(output_path, 'classify_report.jsonl')
    with jsonlines.open(report_file, 'w') as writer:
        for label in labels:
            label_report = {label: report[label]}
            writer.write(label_report)
    print("Classification report saved at:", report_file)
    print("\nClassification Report:")
    print(report)
    return conf_mat

def matrix_general(y_true, y_pred, labels):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    report_file = os.path.join(output_path, 'general_classify_report.jsonl')
    with jsonlines.open(report_file, 'w') as writer:
        for label in labels:
            label_report = {label: report[label]}
            writer.write(label_report)
    print("General classification report saved at:", report_file)
    print("\nGeneral Classification Report:")
    print(report)
    return conf_mat

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


def main(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    #for all labels
    y_true, y_pred, unique_labels = process_dataset(dataset_path)
    result_matrix = matrix(y_true, y_pred, unique_labels)

    # evaluate mistakes and generate JSONL report
    evaluate(result_matrix, unique_labels, output_path, "all_evaluation.json")
    row_mistakes(y_true, y_pred, output_path, 'all_mistakes.json')

    # dataFrame from the result matrix
    conf_df = pd.DataFrame(result_matrix, index=unique_labels, columns=unique_labels)
    plt.figure(figsize=(20, 12))
    sns.heatmap(conf_df, annot=True, fmt='d', annot_kws={'fontsize': 10}, xticklabels=unique_labels, yticklabels=unique_labels, cmap='summer')
    sns.set(font_scale=0.7)
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.ylabel('True Labels', fontsize=10)
    plt.subplots_adjust(left=0.3, bottom=0.3)
    plot_file = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(plot_file)
    print("Confusion matrix plot saved at:", plot_file)


    #for general labels only
    y_true2, y_pred2, unique_labels2 = process_dataset_general(dataset_path)
    result_matrix_general = matrix_general(y_true2,y_pred2, unique_labels2)

    # evaluate mistakes and generate JSONL report
    evaluate(result_matrix_general, unique_labels2, output_path, "general_eval.json")
    row_mistakes(y_true2, y_pred2 , output_path, "general_mistakes.json")

    # dataFrame from the result matrix
    conf_df_general = pd.DataFrame(result_matrix_general, index=unique_labels2, columns=unique_labels2)
    plt.figure(figsize=(20, 12))
    sns.heatmap(conf_df_general, annot=True, fmt='d', annot_kws={'fontsize': 10}, xticklabels=unique_labels2, yticklabels=unique_labels2, cmap='summer')
    sns.set(font_scale=0.7)
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.ylabel('True Labels', fontsize=10)
    plt.subplots_adjust(left=0.3, bottom=0.3)
    plot_file2 = os.path.join(output_path, 'general_confusion_matrix.png')
    plt.savefig(plot_file2)
    print("General confusion matrix plot saved at:", plot_file2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py <dataset_path> <output_path>")
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
        main(dataset_path, output_path)


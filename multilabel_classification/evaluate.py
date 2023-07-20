import argparse
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
    classification_report, roc_curve, roc_auc_score, auc
)

def evaluate_dataset(dataset_file, output_dir):
    y_true = []
    y_pred = []
    label_names = []

    with open(dataset_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            labels = data['labels']
            classification_output = data['classification_output']

            y_true.append([labels[key] for key in sorted(labels.keys())])
            y_pred.append([classification_output[key] for key in sorted(classification_output.keys())])
            label_names = sorted(labels.keys())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    exact_match_ratio = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    hamming_loss_value = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    f1_measure = f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    multilabel_cm = multilabel_confusion_matrix(y_true, y_pred)

    sys.stdout = open(os.path.join(output_dir, 'output.txt'), 'w')

    print('Exact Match Ratio: {0}'.format(exact_match_ratio))
    print('Hamming loss: {0}'.format(hamming_loss_value))
    print('Precision: {0}'.format(precision))
    print('Recall: {0}'.format(recall))
    print('F1 Measure: {0}'.format(f1_measure))



    plt.figure(figsize=(18, 10))
    num_labels = len(label_names)
    num_rows = (num_labels - 1) // 9 + 1
    num_cols = min(num_labels, 9)

    for i in range(num_labels):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.heatmap(multilabel_cm[i], annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.title(f"{label_names[i]}", fontsize=5)
        plt.xlabel('Predicted Labels', fontsize=7)
        plt.ylabel('True Labels', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # Save the figure as "confusion_matrix.png"


    report = classification_report(y_true, y_pred, target_names=label_names)
    print("Classification Report:")
    print(report)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dataset using sklearn metrics')
    parser.add_argument('dataset_file', type=str, help='Path to the dataset file (JSONL format)')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    evaluate_dataset(args.dataset_file, args.output_dir)

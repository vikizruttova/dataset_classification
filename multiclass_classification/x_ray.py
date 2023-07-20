from lib2to3.pgen2.pgen import DFAState
import os
import sys
from matplotlib import axes, axis, cm
import pandas as pd
import matplotlib.pyplot as plt
import json

#run this program in terminal: python3 x_ray.py dataset.jsonl output_xray/

def count_labels(dataset):
    label_counts = {}
    for labels in dataset:
        if 'labels' in labels:
            for label in labels['labels']:
                label_name = label['label']
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
    return label_counts

def count_labels_main(dataset):
    label_counts = {}
    for labels in dataset:
        if 'labels' in labels:
            for label in labels['labels']:
                label_name = label['label'].split('.')[0]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
    return label_counts

def count_languages(dataset):
    language_counts = {}
    for data in dataset:
        if 'lang' in data:
            language = data['lang']
            language_counts[language] = language_counts.get(language, 0) + 1
    return language_counts

def count_modes(dataset):
    mode_counts = {}
    for data in dataset:
        if 'mode' in data:
            mode = data['mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return mode_counts

def count_labels_by_lang(dataset):
    label_counts = {}
    for data in dataset:
        if 'labels' in data and 'mode' in data and 'lang' in data:
            labels = data['labels']
            mode = data['mode']
            lang = data['lang']
            if lang not in label_counts:
                label_counts[lang] = {}
            if mode not in label_counts[lang]:
                label_counts[lang][mode] = {}
            for label in labels:
                label_name = label['label']
                if label_name not in label_counts[lang][mode]:
                    label_counts[lang][mode][label_name] = 0
                label_counts[lang][mode][label_name] += 1
    return label_counts


def analyse_labels(dataset, output_folder):
    label_counts = count_labels(dataset)
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    counts_sorted = [label_counts[label] for label in sorted_labels]
    percentages = [count / sum(counts_sorted) * 100 for count in counts_sorted]
    plt.figure(figsize=(12, 8))
    ax = plt.bar(sorted_labels, counts_sorted)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution of all categories')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()    
    output_path = os.path.join(output_folder, 'label_occurrences.png')
    plt.savefig(output_path, dpi=300)

    plt.close()

def analyse_labels_main(dataset, output_folder):
    label_counts = count_labels_main(dataset)
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    counts_sorted = [label_counts[label] for label in sorted_labels]
    percentages = [count / sum(counts_sorted) * 100 for count in counts_sorted]
    plt.figure(figsize=(12, 8))
    ax = plt.bar(sorted_labels, counts_sorted)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution of main labels')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'label_occurrences_main.png')
    plt.savefig(output_path, dpi=300)

    plt.close()


def analyse_languages(dataset, output_folder):
    language_counts = count_languages(dataset)
    sorted_languages = sorted(language_counts, key=language_counts.get, reverse=True)
    counts_sorted = [language_counts[language] for language in sorted_languages]
    percentages = [count / sum(counts_sorted) * 100 for count in counts_sorted]
    plt.figure(figsize=(12, 8))
    ax = plt.bar(sorted_languages, counts_sorted)
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.title('Language Distribution')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'language_distribution.png')
    plt.savefig(output_path, dpi=300)

    plt.close()

def analyse_modes(dataset, output_folder):
    mode_counts = count_modes(dataset)
    modes = list(mode_counts.keys())
    counts = list(mode_counts.values())
    percentages = [count / sum(counts) * 100 for count in counts]
    
    plt.figure(figsize=(12, 8))
    ax = plt.bar(modes, counts)
    plt.xlabel('Mode')
    plt.ylabel('Count')
    plt.title('Distribution of Modes')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'mode_distribution.png')
    plt.savefig(output_path, dpi=300)

    plt.close()

def analyse_labels_by_lang(dataset, output_folder):
    label_counts = count_labels_by_lang(dataset)

    color_palette = cm.get_cmap('Set3', len(label_counts))
    for lang_index, lang in enumerate(label_counts):
        for mode in label_counts[lang]:
            sorted_labels = sorted(label_counts[lang][mode], key=label_counts[lang][mode].get, reverse=True)
            counts_sorted = [label_counts[lang][mode][label] for label in sorted_labels]
            percentages = [count / sum(counts_sorted) * 100 for count in counts_sorted]
            plt.figure(figsize=(12, 8))
            ax = plt.bar(sorted_labels, counts_sorted, color=color_palette(lang_index))
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.title(f'Label Distribution - Language: {lang} - Mode: {mode}')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            for i, patch in enumerate(ax.patches):
                height = patch.get_height()
                plt.text(patch.get_x() + patch.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
            plt.tight_layout()
            output_path = os.path.join(output_folder, f'label_distribution_{lang}_{mode}.png')
            plt.savefig(output_path, dpi=300)

            plt.close()


def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def analyze_dataset(dataset_file, output_folder):
    create_output_folder(output_folder)
    
    # Load dataset from file
    with open(dataset_file, 'r') as file:
        dataset = [json.loads(line) for line in file]
    
    analyse_labels(dataset, output_folder)
    analyse_labels_main(dataset, output_folder)
    analyse_languages(dataset, output_folder)
    analyse_modes(dataset, output_folder)
    analyse_labels_by_lang(dataset, output_folder)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please provide the dataset file path and output folder path as command-line arguments.")
    else:
        dataset_file = sys.argv[1]
        output_folder = sys.argv[2]
        analyze_dataset(dataset_file, output_folder)
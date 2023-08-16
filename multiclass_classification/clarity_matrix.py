import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize

def compute_metrics(df, labels):
    precision = df.apply(lambda x: x[x.name] / sum(x), axis=1).round(2)
    recall = df.apply(lambda x: x[x.name] / sum(x), axis=0).round(2)
    accuracy = (df.values.diagonal().sum() / df.values.sum()).round(2)

    df['precision'] = precision
    df.loc['recall'] = recall
    df.loc['recall', 'precision'] = accuracy

    return df

def visualize_metrics(df, labels):
        # Extracting the main matrix, precision, recall, and accuracy
    main_matrix = df[labels].iloc[:len(labels)].astype(int)
    precision_values = df[['precision']].iloc[:len(labels)]
    recall_values = df.loc[['recall'], labels]
    accuracy_value = pd.DataFrame([[df.loc['recall', 'precision']]], columns=[''], index=['accuracy'])

    # Setup gridspec for visualization
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], wspace=0.2, hspace=0.2)


    colors = ["#D31B40","#1BD3AE"]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
    cm_cmap =LinearSegmentedColormap.from_list("custom",["#F0F6FD","#2D83E4"])
    norm = Normalize(vmin=(1/labels), vmax=1)
    

    ax0 = plt.subplot(gs[0])
    sns.heatmap(main_matrix, annot=True, fmt="d", cmap=cm_cmap, cbar=False, ax=ax0)
    ax0.set_title("Ground Truth", fontsize=12, weight='bold')
    ax0.set_ylabel("Predicted Label", fontsize=12, weight='bold')
    ax0.xaxis.set_ticks_position('top')

    ax1 = plt.subplot(gs[2])
    sns.heatmap(recall_values, annot=True, fmt=".2f", cmap=custom_cmap, norm=norm, cbar=False, ax=ax1, yticklabels=False)
    ax1.set_xlabel("")
    ax1.set_xticks([])
    ax1.set_ylabel("Recall")
    ax1.set_yticks([])

    ax2 = plt.subplot(gs[1])
    sns.heatmap(precision_values, annot=True, fmt=".2f", cmap=custom_cmap, norm=norm, cbar=False, ax=ax2, xticklabels=False)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_xlabel("Precision")
    ax2.xaxis.set_label_position('top')

    ax3 = plt.subplot(gs[3])
    sns.heatmap(accuracy_value, annot=True, fmt=".2f", cmap=custom_cmap, norm=norm, cbar=False, ax=ax3, xticklabels=False)
    ax3.set_xlabel("Accuracy")
    ax1.set_xticks([])
    ax2.set_ylabel("")
    ax3.set_yticks([])
    ax3.xaxis.set_label_position('top')

    plt.setp(ax0.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.suptitle('"Gerulata Clarity Matrix"', fontsize=15, y=1.07)
    return fig

def generate_clarity_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df = compute_metrics(df, labels)
    fig = visualize_metrics(df, labels)
    return fig

if __name__ == "__main__":
    generate_clarity_matrix(y_true, y_pred, labels)

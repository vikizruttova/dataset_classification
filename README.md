# 🛠️ Tool for Dataset Classification

This GitHub project provides a versatile set of tools for classifying datasets across different classification scenarios. Whether you're dealing with multiclass, multilabel, or binary classification, these scripts help you streamline the process and evaluate the results effectively.

## Multiclass Classification

📊 **x_ray.py** - This script offers a comprehensive analysis of your dataset. It generates informative graphs showcasing modes, languages, and label distributions.

🔍 **classify.py** - With this script, you can classify entries in a JSONL dataset using an API for hate speech detection. It then returns an adjusted dataset, enhancing its classification accuracy.

📈 **evaluate.py** - Evaluate your adjusted dataset using this script. It generates confusion matrices, the classification reports, and identifies misclassifications. Moreover, it introduces the concept of clarity matrices, which provide a specialized evaluation by allowing you to split datasets based on chosen metrics (e.g., [lang], [lang, mode]) and subsequently generate evaluation metrics and ROC AUC curves for each split.

## Multilabel Classification

📊 **x_ray.py** - Similar to its usage in multiclass classification, this script aids in visualizing modes, languages, and label distributions in your dataset.

🔍 **classify.py** - Extend the classification capabilities to multilabel scenarios using this script. It employs an API for hate speech detection to adjust your dataset effectively.

📊 **evaluate.py** - After adjusting the dataset, utilize this script to produce confusion matrices, classification reports, and pinpoint misclassifications.

🔄 **transformer.py** - For multilabel evaluation, this script transforms the dataset by creating a binary representation for each label.

## Binary Classification

🔍 **classify.py** - This script is tailored for binary classification. By leveraging a hate speech detection API, it assigns each entry as either hate speech or non-hate speech, resulting in an enhanced dataset.

📊 **evaluate.py** - Once your dataset is adjusted, employ this script to generate insightful confusion matrices, classification reports, and ROC curves for comprehensive evaluation.

Feel free to explore and adjust these scripts for your dataset classification needs. They are designed to simplify your workflow and provide clear insights into the classification process. Your feedback and contributions are greatly appreciated! 🚀








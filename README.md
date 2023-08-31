# 🛠️ Tool for Dataset Classification

This GitHub project provides a versatile set of tools for classifying datasets across different classification scenarios. Whether you're dealing with multiclass, multilabel, or binary classification, these scripts help you streamline the process and evaluate the results effectively.

## Multiclass Classification 💡

📊 **x_ray.py** - This script offers a comprehensive analysis of your dataset. It generates informative graphs showcasing modes, languages, and label distributions.

🔍 **classify.py** - With this script, you can classify entries in a JSONL dataset using an API for hate speech detection. It then returns an adjusted dataset, enhancing its classification accuracy.

📈 **evaluate.py** - Evaluate your adjusted dataset using this script. It generates confusion matrices, the classification reports, and identifies misclassifications. Moreover, it introduces the concept of clarity matrices, which provide a specialized evaluation. Additionally, you are able to split datasets based on chosen metrics _(e.g., [lang], [lang, mode])_ and subsequently generate evaluation metrics and ROC AUC curves for each split.

<p align="center">
    <strong>The Clarity Matrix</strong>
  <br>
  <img width="670" alt="Screenshot 2023-08-31 at 12 31 15" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/7e00cffc-b3d5-4095-bcf8-00a002ce3754">
</p>

<strong>The ROC curves for each label in the whole dataset</strong>
  <br>
<p align="center">
  <img width="1227" alt="Screenshot 2023-08-31 at 12 31 27" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/0625cb04-0e27-47c1-924e-cb2f1c9de70b">
</p>



## Multilabel Classification 💡

📊 **x_ray.py** - Similar to its usage in multiclass classification, this script aids in visualizing modes, languages, and label distributions in your dataset.

<p align="center">
  <img width="1005" alt="Screenshot 2023-08-31 at 12 50 37" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/af108fb2-5b0c-4cac-a978-7063664da302">
</p>

🔍 **classify.py** - Extend the classification capabilities to multilabel scenarios using this script. It employs an API for hate speech detection to adjust your dataset effectively.

📊 **evaluate.py** - After adjusting the dataset, utilize this script to produce confusion matrices, classification reports, and pinpoint misclassification.

<p align="center">
  <img width="1197" alt="Screenshot 2023-08-31 at 12 52 14" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/19e69014-824d-4f9c-9f75-5c9e87dd3dce">
</p>

🔄 **transformer.py** - For multilabel evaluation, this script transforms the dataset by creating a binary representation for each label.

## Binary Classification 💡

📊 **x_ray.py** - Similar to its usage in multiclass classification, this script aids in visualizing modes, languages, and label distributions in your dataset.

<p align="center">
  <img width="1004" alt="Screenshot 2023-08-31 at 12 55 53" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/5f32bb89-4081-4c2f-936d-7cd0da17867c">
</p>

🔍 **classify.py** - This script is tailored for binary classification. By leveraging a hate speech detection API, it assigns each entry as either hate speech or non-hate speech, resulting in an enhanced dataset.

📊 **evaluate.py** - Once your dataset is adjusted, employ this script to generate insightful confusion matrices, classification reports, and ROC curves for comprehensive evaluation.

<p align="center">
  <img width="684" alt="Screenshot 2023-08-31 at 12 53 26" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/64d1f271-646d-438b-9e62-b7670954247a">
  <img width="667" alt="Screenshot 2023-08-31 at 12 53 31" src="https://github.com/vikizruttova/dataset_classification/assets/72280373/69735d6a-55c5-4041-b64c-58dabb3dc36d">
  
</p>

Feel free to explore and adjust these scripts for your dataset classification needs. They are designed to simplify your workflow and provide clear insights into the classification process. Your feedback and contributions are greatly appreciated! 🚀








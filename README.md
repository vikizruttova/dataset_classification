# Tool for dataset classification


## Multiclass classification

**x_ray.py** - takes a look at the dataset and returns graphs: modes, languages, distribution of labels

**classify.py** - takes JSONL dataset and each entry is classified in API for hate speech detection and returns adjusted dataset

**evaluate.py** - takes adjusted dataset from classify.py, creates confusion matrices, classification reports, and mistakes in classification.

## Multilabel classification

**x_ray.py** - takes a look at the dataset and returns graphs: modes, languages, distribution of labels

**classify.py** - takes JSONL dataset and each entry is classified in API for hate speech detection and returns adjusted dataset

**evaluate.py** - takes adjusted dataset from classify.py, creates confusion matrices, classification reports, and mistakes in classification.

**transformer.py** - adjust the dataset for multilabel evaluation - creates a binary representation of each label.


## Binary classification

**classify.py** - takes JSONL dataset and each entry is classified in API for hate speech detection and returns adjusted dataset - hate-speech or not(other)

**evaluate.py** - takes adjusted dataset from classify.py, creates confusion matrices, classification reports, and roc curve.


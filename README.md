# Tool for dataset classification - hate speech

**x_ray.py** - takes a look at the dataset and returns graphs: modes, languages, distribution of labels

**classify.py** - takes JSONL dataset and each entry is classified in API for hate speech detection and returns adjusted dataset

**evaluate.py** - takes adjusted dataset from classify.py, creates confusion matrices, classification reports, and mistakes in classification.

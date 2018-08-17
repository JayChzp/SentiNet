# SentiNet
The code for Encoding Sentiment Information into Word Vectors for Sentiment Analysis

# Requirment
* Python = 3.6.4
* Pytorch = 0.3.0
* Numpy >= 1.14.0
* nltk
* scipy

# Dataset
Put the dataset proprocessed by code of [https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence) in the directory of `dataset/MR/`. 

Download GoogleNews-vectors-negative300.bin in the directory of the code.

# Run

```
python sentinet.py --mr --embedding_path GoogleNews-vectors-negative300.bin --isBinary
```

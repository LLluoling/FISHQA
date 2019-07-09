# FISHQA ( Financial-Sentiment-Analysis-with-Hierarchical-Query-driven-Attention )
### This is a Tensorflow implementation of [Beyond Polarity: Interpretable Financial Sentiment Analysis with Hierarchical Query-driven Attention](https://www.ijcai.org/proceedings/2018/0590.pdf)

## Requirements
 * python 3.6.1
 * Tensorflow 1.11.0
 * jieba 0.39

## Code Introduction 

### Step 1: Preprocess data
```bash
python preprocess_train.py 
python preprocess_test.py
```
Preprocess training dataset/test dataset.
Remember to modify the dictionary, fiterwords based on your own datasets.

### Step 2: Training model
```bash
python train.py
```
```bash
cd FISHQA/code
```
Set params based on your own datasets and train you own model

### Step 3: Test model
```bash
python test.py 
```

### Step 4: Simple attention visualization 
```bash
python view.py 
```


## Data Introduction 
* Modify your own queries(FISHQA/Query) based on your own datasets and prior knowledge. Each `query` can be manually decided.
* Notice that under folder `temp/` is a subset of our preprocessed data.
* As the our dataset is private, we cannot release it. We put two raw samples in folder `train_data` and `test_data` individually.
* Under folder `dictionary/`, there are some extra dictionaries summarized by professional for Chinese financial news.

# README

## Introduction

This repository contains the code and resources used in a study aimed at developing a model for classifying tweets into offensive language, hate speech, or neither. The model leverages Natural Language Processing (NLP), Recurrent Neural Networks (RNNs), and machine learning to automatically categorize tweets based on their content.

## Repository Contents

- `Model.ipynb`: Jupyter notebook containing the code for the models.
- `train.csv`: Dataset used for training the models.
- `model_evaluation_results.xlsx`: Excel file containing the evaluation results of the models.
- `ModelArch.jpg`: Diagram illustrating the architecture of the models.
- Confusion Matrices: 
  - `model_GRU.png`
  - `model_LSTM.png`
  - `model_LSTM_ConvGRU.png`
  - `model_LSTM_ConvGRU_word2vec.png`
  - `model_LSTM_GRU.png`
  - `model_LSTM_GRU_Word2Vec.png`

## Methodology

### Data Preprocessing

The preprocessing stage involves several steps:

1. Conversion of tweets to lowercase to ensure uniformity and avoid duplication based on case differences.
2. Fitting a tokenizer on the tweets, which turns each text into a sequence of integers (each integer being the index of a token in a dictionary). The sequences are padded to ensure that all sequences in a list have the same length.
3. Conversion of string labels to one-hot encoding. This process converts each categorical value into a new categorical value and assigns a binary value of 1 or 0.
4. Splitting the data into training and testing sets using the `train_test_split` function from the sklearn library.

### Embeddings

Word2Vec and GloVe are two popular methods used in NLP to convert words into numerical vectors. These methods are based on the idea that the meaning of a word can be inferred by the context in which it appears. In this study, both methods of embeddings were tried, but the GloVe version was used for the final results.

### Model Architecture


Two main models and two baseline models were trained and tested on the dataset:
![ModelArch](https://github.com/SeherD/ToxicLanguageDetection/assets/59703840/a3795806-2f80-4c68-ae75-ce40b14d54e7)


1. LSTM-GRU Model: A sequential model that leverages the power of both Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) layers.
2. LSTM-ConvGru Model: A hybrid model that combines the strengths of Convolutional Neural Networks (CNNs) and RNNs for sequence processing.
3. LSTM Model: A baseline model similar to the LSTM-GRU model but without a GRU layer.
4. GRU Model: A baseline model similar to the LSTM-GRU model but without an LSTM layer.

### Evaluation Metrics

The following evaluation metrics were used for training and testing the models:

- Categorical Cross Entropy Loss
- Categorical Accuracy
![Accuracy](https://github.com/SeherD/ToxicLanguageDetection/assets/59703840/a2ef3995-997a-49fe-83f9-bc78d2e70fd0)

- Categorical AUC (Area Under the Curve)
- Precision
- Recall
- F1-Score
![F1-Score - LSTM_ConvGRU GloVE vs Word2Vec](https://github.com/SeherD/ToxicLanguageDetection/assets/59703840/41de995d-2c94-401b-8f18-d8ba07488315)


Confusion Matrices were also generated for all models. 
![model_LSTM_GRU](https://github.com/SeherD/ToxicLanguageDetection/assets/59703840/b3bd6ee1-d186-4d5f-a10a-bb616ccf92d9)
![model_LSTM_ConvGRU](https://github.com/SeherD/ToxicLanguageDetection/assets/59703840/382209f2-5797-4a8f-8ec1-457543598cf7)

## Dataset

The selected dataset, titled "Hate Speech and Offensive Language Detection", was collected using Twitter's public API under search terms containing hate speech or offensive language. The dataset is provided in .csv format and contains annotations provided by multiple annotators for each tweet. The dataset can be accessed [here](https://www.kaggle.com/datasets/thedevastator/hate-speech-and-offensive-language-detection?resource=download).

## Type of Data

The dataset contains the following fields:

- `count` (numerical): Total number of annotations in the tweet
- `hate_speech_count` (numerical): How many annotations classified a particular tweet as hate speech. Hate speech is mainly presented in the data as racism, homophobia, ableism, and sexism. Most slurs are considered hate speech. 
- `offensive_language_count` (numerical): Denotes how many annotations classified a particular tweet as offensive language speech. Offensive language is mainly classified by the presence of curse words in the tweet. 
- `neither_count` (numerical): How many annotations classified a particular tweet as neither hate speech nor offensive language.
- `class` (numerical): Denotes the classification assigned by the most annotations. In other words, a numerical label depicting class that had the highest value among the hate_speech_count, offensive_language_count, and neither_count: A value of 0 means the tweet is classified as hate speech, 1 indicates offensive language, and 2 means the tweet is classified as neither hate speech nor offensive language. 
- `tweet` (text): the written content of the tweet

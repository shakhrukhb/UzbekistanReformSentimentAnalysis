# Sentiment Analysis on Social Media Comments: Evaluating Public Perception for Reforms in Uzbekistan

## Overview
Over the last seven years, Uzbekistan has undergone significant reforms in areas such as governance, economics, and social policy. This project aims to analyze public sentiment using news articles using more than 23K posts with 700K comments extracted from kun.uz instagram profile. This analysis would be beneficial for the areas of reforms to determine what areas are well-received and those that need more refinement of reforms. My primary research question is: "How does the online community perceive the reforms implemented in Uzbekistan?" This project includes two primary components: Neural Network-based text classification and traditional Machine Learning-based text classification. It is designed to classify text comments into various sentiment categories: Positive, Neutral, and Negative and in order to train the models, 10K comments manually labelled. So, the project has classification.py, nn.py, ml.py and preprocessing.py scripts.

## Requirements
- Python 3.x
- Libraries as listed in `requirements.txt`

## Installation
To replicate the results, you should download the Excel files from this link: `https://drive.google.com/drive/folders/186tgd1DovveADNlHM_6KIwtQh9HtxYIu?usp=drive_link`
After downloading, you should put the files into Data folder. Note that I strongly recommend to use virtual environment when installing libraries.
To create and install the necessary libraries, run:
```bash
python -m venv venv # creates virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Preprocesing
The preprocess.py script handles text data preprocessing, including handling NA values, conversion from Latin to Cyrillic text, removing emojis, handling URLs, and some necessary text cleaning.

## Classification
In the classification.py file includes classifying the posts into categories into like Society, Health, Sports, World, Science and Technology, Politics, Crime, Economy, Culture, Show Business and Miscellaneous using `coppercitylabs/uzbek-news-category-classifier` pre-trained model.

## Neural Network Models
Neural Network models are included in nn.py file. These models are implemented using Transformers including architectures like CNN, LSTM, GRU, and a baseline feed-forward network. Note that in Train function, you should specify the model architecture. To train the model with a selected Neural Network model, run:
```bash
python nn.py train
```

Feed Forward = f1_score: 0.6419, accuracy: 0.7652, precision: 0.6712, recall: 0.5793
LTSM = f1_score: 0.6188, accuracy: 0.7454, precision: 0.6190, recall: 0.6149
GRU = f1_score: 0.6346, accuracy: 0.7573, precision: 0.6437, recall: 0.6090
CNN = f1_score: 0.6682, accuracy: 0.7836, precision: 0.6958, recall: 0.6230

To make predictions with a trained Neural Network model, run:
```bash
python nn.py predict
```

## Machine Learning models
Machine Learning models are included in ml.py file. These models are implemented using sklearn including architectures like Logistic Regression, Random Forest and KNN. Note that in Train and Predict functions, you should specify the model_type. I used 5-fold CV for Machine Learning models. To train the model with a selected Machine Learning model, run:
```bash
python ml.py train
```
LR = Accuracy: 0.6105, Precision: 0.6793, Recall: 0.5702, F1_score: 0.5948
RF = Accuracy: 0.5979, Precision: 0.64345, Recall: 0.5687, F1_score: 0.5872
KNN = Accuracy: 0.4692, Precision: 0.6361, Recall: 0.3970, F1_score: 0.3346

To make predictions with a trained Machine Learning model, run:
```bash
python ml.py predict
```

## Vizualizations
All vizualizations are included nlp.ipynb jupyter notebook file.
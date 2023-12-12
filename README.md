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
Neural Network models are included in nn.py file. These models are implemented using Transformers including architectures like CNN, LSTM, GRU, and a baseline feed-forward network. Note that in Train function, you should specify the model architecture. The higherst F1 score achieved for Feed-forward = 0.647, for CNN = 0.651, for LTSM = 0.597, for GRU = 0.634. To train the model with a selected Neural Network model, run:
```bash
python nn.py train
```
To make predictions with a trained Neural Network model, run:
```bash
python nn.py predict
```

## Machine Learning models
Machine Learning models are included in ml.py file. These models are implemented using sklearn including architectures like Logistic Regression, Random Forest and KNN. Note that in Train and Predict functions, you should specify the model_type. I used 5-fold CV for Machine Learning models. The average accuracy achieved for LR = 0.594, RF = 0.581 and knn = 0.464. To train the model with a selected Machine Learning model, run:
```bash
python ml.py train
```

To make predictions with a trained Machine Learning model, run:
```bash
python ml.py predict
```

## Vizualizations
All vizualizations are included nlp.ipynb jupyter notebook file.

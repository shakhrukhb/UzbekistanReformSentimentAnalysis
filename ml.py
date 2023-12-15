import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from joblib import dump, load
from preprocess import latin_cyrillic

def vectorize(df, column_name='Comment_ru'):
    vectorizer = TfidfVectorizer(max_features=20000)
    X = vectorizer.fit_transform(df[column_name].astype(str)).toarray()
    return X, vectorizer

def train(file_path=("Data/Comments_training.xlsx", 'Comment'), model_path="model", model_type='knn'):
    df = latin_cyrillic(file_path)
    df['Labels'] = df['Labels'].astype(int)
    X, vectorizer = vectorize(df, file_path[1])
    y = df['Labels']
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Invalid model type")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        print(f'Accuracy: {np.mean(accuracies)}, Precision: {np.mean(precisions)}, Recall: {np.mean(recalls)}, F1_score: {np.mean(f1s)}')

    # Fit the model on the entire dataset
    model.fit(X, y)

    model_folder = os.path.join(model_path, model_type)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save model and vectorizer
    dump(model, os.path.join(model_folder, f"{model_type}.joblib"))
    dump(vectorizer, os.path.join(model_folder, 'vectorizer.joblib'))

# LR = Accuracy: 0.6104923632459863, Precision: 0.6792637944260306, Recall: 0.5702218590720245, F1_score: 0.5947710714345178
# RF = Accuracy: 0.5979480310349876, Precision: 0.643473915593359, Recall: 0.5687423658787203, F1_score: 0.5872455186633763
# KNN = Accuracy: 0.46922998096911145, Precision: 0.6360534273118873, Recall: 0.39697323259738, F1_score: 0.3345738484524795

def predict(input_path=("Data/Comments.xlsx", 'Comment'), model_path="model", model_type="logistic_regression"):
    model_folder = os.path.join(model_path, model_type)
    model = load(os.path.join(model_folder, f"{model_type}.joblib"))
    vectorizer = load(os.path.join(model_folder, "vectorizer.joblib"))

    df = latin_cyrillic(input_path)
    X = vectorizer.transform(df[input_path[1]].astype(str)).toarray()
    predictions = model.predict(X)

    df['Predicted_Label'] = predictions
    output_file = "Data/Comments_predictions_ml.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description=" Comment Classification script")
    parser.add_argument("command", choices=["train", "predict"], help="train or predict")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save or load the model")
    parser.add_argument("--file_path", type=str, nargs=2, default=("Data/Comments_training.xlsx", 'Comment'), help="Path to the training dataset")
    parser.add_argument("--input_path", type=str, nargs=2, default=("Data/Comments.xlsx", 'Comment'), help="Path to the input data for prediction")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(file_path=args.file_path, model_path=args.model_path)
    elif args.command == "predict":
        predict(input_path=args.input_path, model_path=args.model_path)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datasets
from transformers import AutoTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from preprocess import latin_cyrillic

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("coppercitylabs/uzbert-base-uncased")

def tokenize(examples):
    """Converts text to 'input_ids'"""
    return tokenizer(examples["Comment_ru"], truncation=True, max_length=64, padding="max_length")

def gather_labels(example):
    """Extracts label data"""
    labels = ['Positive', 'Neutral', 'Negative']
    return {"labels": [float(example[l]) for l in labels]}

def cnn(vocab_size, embedding_dim=256, num_filters=256, kernel_size=5, dropout_rate=0.3, num_labels=3, learning_rate=0.00005):
    """Defines a CNN model for text classification."""
    input_layer = Input(shape=(64,), dtype=tf.int32, name="input_ids")
    embedded_sequences = Embedding(vocab_size, embedding_dim)(input_layer)
    x = Conv1D(num_filters, kernel_size, activation='relu')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.F1Score(average="macro"), BinaryAccuracy(), Precision(), Recall()])
    return model

def ltsm(vocab_size, embedding_dim=256, lstm_units=256, dropout_rate=0.3, num_labels=3, learning_rate=0.00005):
    """Defines an RNN (LSTM) model for text classification."""
    input_layer = Input(shape=(64,), dtype=tf.int32, name="input_ids")
    embedded_sequences = Embedding(vocab_size, embedding_dim)(input_layer)
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(embedded_sequences)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.F1Score(average="macro"), BinaryAccuracy(), Precision(), Recall()])
    return model

def gru(vocab_size, embedding_dim=256, gru_units=256, dropout_rate=0.3, num_labels=3, learning_rate=0.001):
    """Defines an RNN (GRU) model for text classification."""
    input_layer = Input(shape=(64,), dtype=tf.int32, name="input_ids")
    embedded_sequences = Embedding(vocab_size, embedding_dim)(input_layer)
    x = tf.keras.layers.GRU(gru_units, return_sequences=False)(embedded_sequences)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.F1Score(average="macro"), BinaryAccuracy(), Precision(), Recall()])
    return model

def feed_forward(vocab_size, embedding_dim=256, dense_units=256, dropout_rate=0.3, num_labels=3, learning_rate=0.0001):
    """Defines a baseline feed-forward neural network model for text classification."""
    input_layer = Input(shape=(64,), dtype=tf.int32, name="input_ids")
    embedded_sequences = Embedding(vocab_size, embedding_dim)(input_layer)
    x = tf.keras.layers.Flatten()(embedded_sequences)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.F1Score(average="macro"), BinaryAccuracy(), Precision(), Recall()])
    return model

def train(model_path="model", file_path=("Data/Comments_training.xlsx", 'Comment'), dev_split=0.15):
    """Train the model."""
    df = latin_cyrillic(file_path)
    df['Comment_ru'] = df['Comment_ru'].astype(str).str.lower()
    df['Positive'] = (df['Labels'] == 1).astype(int)
    df['Neutral'] = (df['Labels'] == 0).astype(int)
    df['Negative'] = (df['Labels'] == 2).astype(int)
    df = df[['Comment_ru', 'Positive', 'Neutral', 'Negative']]
    train_df, dev_df = train_test_split(df, test_size=dev_split)
    
    train_dataset = datasets.Dataset.from_pandas(train_df)
    dev_dataset = datasets.Dataset.from_pandas(dev_df)
    
    train_dataset = train_dataset.map(gather_labels).map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(gather_labels).map(tokenize, batched=True)

    train_dataset = train_dataset.to_tf_dataset(columns='input_ids', label_cols='labels', batch_size=12, shuffle=True)
    dev_dataset = dev_dataset.to_tf_dataset(columns='input_ids', label_cols='labels', batch_size=12)
    
    # Specify the model:
    model = feed_forward(tokenizer.vocab_size) # f1_score: 0.6419, accuracy: 0.7652, precision: 0.6712, recall: 0.5793
    # model = ltsm(tokenizer.vocab_size)         # f1_score: 0.6188, accuracy: 0.7454, precision: 0.6190, recall: 0.6149
    # model = gru(tokenizer.vocab_size)          # f1_score: 0.6346, accuracy: 0.7573, precision: 0.6437, recall: 0.6090
    # model = cnn(tokenizer.vocab_size)          # f1_score: 0.6682, accuracy: 0.7836, precision: 0.6958, recall: 0.6230
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=3, mode='max', verbose=1, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_f1_score', mode='max', save_best_only=True)
    model.fit(train_dataset, epochs=15, validation_data=dev_dataset, callbacks=[model_checkpoint, early_stopping])
    model.save(model_path)

def predict(model_path="model", input_path=("Data/Comments.xlsx", 'Comment')):
    """Generate predictions using the trained model."""
    # Load dataset and convert from Latin to Cyrillic
    df = latin_cyrillic(input_path)
    df['Comment_ru'] = df['Comment_ru'].astype(str).str.lower()  # Ensure column name matches with input_path
    df = df[['Comment_ru', 'URL']]

    # Convert to Hugging Face dataset and tokenize
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert to TensorFlow dataset
    tf_dataset = hf_dataset.to_tf_dataset(columns="input_ids", batch_size=12)

    # Load model and generate predictions
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(tf_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    # Map predicted labels back to original values
    label_map = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    df['Predicted_Label'] = [label_map[label] for label in predicted_labels]

    # Save the predictions to a Excel file
    output_file = "Data/Comments_predictions.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return df

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
        train(model_path=args.model_path, file_path=args.file_path)
    elif args.command == "predict":
        predict(model_path=args.model_path, input_path=args.input_path)

from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
from preprocess import latin_cyrillic

# Initialize the classifier pipeline
classifier = pipeline('text-classification', model='coppercitylabs/uzbek-news-category-classifier')

# Load the dataset and convert from Latin to Cyrillic
file_path = ("Data/Posts.xlsx", 'Posts')
df = latin_cyrillic(file_path)

# Define a function to classify text with error handling
def classify_text(text, chunk_size=512):
    try:
        # If the text is longer than the chunk size, split it into smaller parts
        if len(text) > chunk_size:
            # Split the text into chunks
            parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            # Classify each part and keep the most common prediction and score
            predictions = [classifier(part)[0] for part in parts]
            # Aggregate the results
            labels = [pred['label'] for pred in predictions]
            scores = [pred['score'] for pred in predictions]
            # Use a simple strategy like taking the mode of the labels and average of the scores
            prediction_label = max(set(labels), key=labels.count)
            prediction_score = sum(scores) / len(scores)
        else:
            prediction = classifier(text)[0]
            prediction_label = prediction['label']
            prediction_score = prediction['score']
        return prediction_label, prediction_score
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Add a progress bar for the apply function
tqdm.pandas(desc="Classifying")

# Apply the classifier to the Post_ru column
df[['Category', 'Confidence']] = pd.DataFrame(df['Posts_ru'].progress_apply(classify_text).tolist(), index=df.index)

# Save the classified results to a new Excel file
df.to_excel('Data/Classified_Posts.xlsx', index=False)

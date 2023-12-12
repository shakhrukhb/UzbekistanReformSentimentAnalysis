import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from UzTransliterator import UzTransliterator

# Compile a regex pattern for matching emojis
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)

# Create an instance of the UzTransliterator class
transliterator = UzTransliterator.UzTransliterator()

def latin_cyrillic(file_info):
    file_path, column_name = file_info
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
        
    def remove_emojis(text):
        return emoji_pattern.sub(r'', text)

    # Regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def contains_url(text):
        if pd.isna(text) or not isinstance(text, str):
            return False  # If the text is NaN or not a string, it doesn't contain a URL
        return url_pattern.search(text) is not None

    def convert_urls_to_text(df, url_column):
        # Convert the URL column to strings to ensure all data is text
        df[url_column] = '`' + df[url_column].astype(str)
        return df

    # Define a function to replace non-breaking spaces with regular spaces and then transliterate
    def replace_and_transliterate(text):
        if not isinstance(text, str):
            return None
        text = remove_emojis(text).replace(u' ', ' ')  # Remove emojis
        # Split the text into words and transliterate each word individually
        words = text.split()
        transliterated_words = []
        errors = []  # To collect errors
        for word in words:
            if 'w' in word or 'W' in word:
                transliterated_words.append(word)
            else:
                try:
                    transliterated_words.append(transliterator.transliterate(word, from_="lat", to="cyr"))
                except IndexError:
                    errors.append(word)
        if errors:
            print(f"Error transliterating words: {errors}")
            return None
        return ' '.join(transliterated_words)

    # Drop rows that contain a URL
    mask = df[column_name].apply(contains_url)
    df = df[~mask]

    # Apply the transliteration function and drop rows causing IndexError
    tqdm.pandas(desc=f"Transliterating rows in {os.path.basename(file_path)}")
    df[column_name + '_ru'] = df[column_name].progress_apply(lambda text: replace_and_transliterate(str(text)))
        
    # Drop rows with None
    df = df.dropna(subset=[column_name + '_ru'])
    df = convert_urls_to_text(df, 'URL')

    return df

def preprocess_hashtags(file_path):
    
    # Load and preprocess data
    data = pd.read_excel(file_path)
    data = data.drop_duplicates(subset='URL')

    data['Post'] = data['Title'].fillna('') + " " + data['Content'].fillna('')
    data['Post'] = data['Post'].fillna('').apply(str)

    # Define a nested function to split hashtags
    def split_hashtags(text):
        if pd.isna(text) or text == "":
            return []
        text = text.replace('\u200b', '')  # Remove zero-width spaces
        # Split by '# ' and then by '#' in case there are hashtags without a space
        return [tag.strip() for part in text.split('# ') for tag in part.split('#') if tag.strip()]

    # Apply the split_hashtags function to the 'Hashtags' column
    data['Hashtags'] = data['Hashtags'].apply(split_hashtags)

    # Drop rows with NaN or empty lists in 'Hashtags' column
    data = data.dropna(subset=['Hashtags'])
    data = data[data['Hashtags'].map(len) > 0]
    return data



import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize stemmer and stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Clean text by:
    - Lowercasing
    - Removing non-alphabetic characters
    - Removing stopwords
    - Applying stemming
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    try:
        df = pd.read_csv('data/fake_news_dataset.csv', encoding='utf-8')
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Drop the 'date' column due to high NaT percentage
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)
        print("🗑️ Dropped 'date' column due to high missing values.")

    # Fill missing 'source' and 'author' with 'Unknown'
    for col in ['source', 'author']:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna('Unknown')
            missing_after = df[col].isnull().sum()
            print(f"✅ Filled {missing_before} missing values in '{col}' with 'Unknown'. Remaining: {missing_after}")

    # Encode 'label' column ('fake' -> 0, 'real' -> 1)
    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower().map({'fake': 0, 'real': 1})
        if df['label'].isnull().any():
            print("⚠️ Warning: Some labels could not be mapped. Check for inconsistencies.")
        else:
            print("✅ Label encoding completed.")

    # Drop rows with missing labels or text (if any)
    before_drop = df.shape[0]
    df.dropna(subset=['text', 'label'], inplace=True)
    after_drop = df.shape[0]
    print(f"✅ Dropped {before_drop - after_drop} rows with missing 'text' or 'label'.")

    # Apply text cleaning to 'text' and 'title'
    print("🧹 Cleaning text data. This may take a moment...")
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_title'] = df['title'].apply(clean_text)
    print("✅ Text cleaning completed.")

    # Save the cleaned dataset
    df.to_csv('data/clean_fake_news.csv', index=False)
    print("✅ Cleaned dataset saved to 'data/clean_fake_news.csv'.")

if __name__ == "__main__":
    main()

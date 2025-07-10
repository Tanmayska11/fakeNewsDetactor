

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse

def main():
    # Load cleaned dataset
    df = pd.read_csv('data/clean_fake_news.csv')
    print("✅ Cleaned dataset loaded.")

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=5,
        max_df=0.9
    )

    # Fit and transform the 'clean_text' column
    X = vectorizer.fit_transform(df['clean_text'])
    print(f"✅ TF-IDF vectorization completed. Shape: {X.shape}")

    # Save the vectorizer for future use during prediction
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("✅ TF-IDF vectorizer saved to 'models/tfidf_vectorizer.pkl'.")

    # Save the TF-IDF features as a sparse matrix for model training
    sparse.save_npz('data/tfidf_features.npz', X)
    print("✅ TF-IDF features saved to 'data/tfidf_features.npz'.")

if __name__ == "__main__":
    main()

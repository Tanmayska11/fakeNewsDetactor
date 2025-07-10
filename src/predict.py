

import pandas as pd
import joblib
from scipy import sparse

def predict_fake_news(text_list):
    """
    Predicts whether the given list of text samples are fake or real.
    Input:
        text_list (list): List of strings (news articles)
    Returns:
        List of predictions ('FAKE' or 'REAL')
    """
    # Load the trained model and vectorizer
    model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    # Transform the input text using the loaded vectorizer
    X = vectorizer.transform(text_list)

    # Make predictions
    preds = model.predict(X)

    # Map numeric predictions to labels
    label_map = {0: 'FAKE', 1: 'REAL'}
    pred_labels = [label_map[p] for p in preds]
    
    return pred_labels

if __name__ == "__main__":
    # Example manual testing
    sample_texts = [
        "Government announces new economic policy to improve stability and growth across all sectors.",
        "Breaking! Celebrity endorses miracle pill that can make you lose weight in 3 days without exercise!"
    ]
    
    predictions = predict_fake_news(sample_texts)
    
    for text, label in zip(sample_texts, predictions):
        print(f"\nText: {text}\nPrediction: {label}")
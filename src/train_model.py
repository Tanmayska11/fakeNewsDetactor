# train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import sparse
import joblib

def main():
    # Load cleaned dataset for labels
    df = pd.read_csv('data/clean_fake_news.csv')
    print("✅ Cleaned dataset loaded.")

    # Load TF-IDF features
    X = sparse.load_npz('data/tfidf_features.npz')
    print(f"✅ TF-IDF features loaded with shape {X.shape}.")

    # Load labels
    y = df['label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split into train ({X_train.shape[0]}) and test ({X_test.shape[0]}).")

    # Initialize and train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✅ Model training completed.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'models/fake_news_model.pkl')
    print("✅ Trained model saved to 'models/fake_news_model.pkl'.")

if __name__ == "__main__":
    main()

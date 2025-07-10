# save_test_results.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy import sparse
import joblib
import os

def save_test_results():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load cleaned dataset for true labels
    df = pd.read_csv('data/clean_fake_news.csv')
    print("✅ Cleaned dataset loaded.")

    # Load TF-IDF features
    X = sparse.load_npz('data/tfidf_features.npz')
    print(f"✅ TF-IDF features loaded with shape {X.shape}.")

    # Load labels
    y = df['label'].values

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split into train ({X_train.shape[0]}) and test ({X_test.shape[0]}).")

    # Load the trained model
    model = joblib.load('models/fake_news_model.pkl')
    print("✅ Trained model loaded.")

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability for label=1 (REAL)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save metrics to text file
    with open('results/test_results.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}\n")
    print("✅ Test metrics saved to 'results/test_results.txt'.")

    # Save detailed predictions to CSV
    results_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred,
        'Probability_REAL': y_proba
    })
    results_df.to_csv('results/test_predictions.csv', index=False)
    print("✅ Detailed predictions saved to 'results/test_predictions.csv'.")

def main():
    save_test_results()

if __name__ == "__main__":
    main()

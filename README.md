# 📰 Fake News Detector - Internship Project

## 📌 Overview

This **Fake News Detector** is a professional end-to-end machine learning pipeline built for your internship to practice **text classification, data cleaning, feature engineering, model training, and Streamlit deployment**.

It uses:  
✅ **TF-IDF Vectorization**  
✅ **Logistic Regression Classifier**  
✅ **Streamlit App for Testing**  
✅ **Modular, clean code structure** for real-developer workflow learning.

⚠️ **Disclaimer:** This project is for **learning and demonstration only**. The model currently achieves ~50% accuracy on the training dataset and is **not ready for real-world fake news detection.**

---

## 📂 Project Structure

fake_news_detector/
├── data/
│ ├── fake_news.csv # Raw dataset
│ ├── clean_fake_news.csv # Cleaned dataset
│ └── tfidf_features.npz # Saved TF-IDF features
│
├── models/
│ ├── fake_news_model.pkl # Trained model
│ └── tfidf_vectorizer.pkl # Saved vectorizer
│
├── src/
│ ├── data_exploration.py # Initial exploration
│ ├── data_cleaning.py # Data cleaning pipeline
│ ├── feature_engineering.py # TF-IDF vectorization
│ ├── train_model.py # Model training and evaluation
│ └── predict.py # Local prediction utility
│ └── save_test_results.py # Script to evaluate and save test results
│
├── app/
│ └── streamlit_app.py # Streamlit app for testing
│
└── README.md # Project documentation



---

## 🚀 Features Implemented

✅ **Data Exploration:** Shape, nulls, duplicates, distributions.  
✅ **Data Cleaning:** Text cleaning, label encoding, null handling.  
✅ **Feature Engineering:** TF-IDF vectorization with n-grams.  
✅ **Model Training:** Logistic Regression with evaluation metrics.  
✅ **Prediction:** Local testing with `predict.py`.  
✅ **Streamlit Deployment:** User-friendly interface for testing articles interactively.  
✅ **Automated Test Results Saving:** Integration of test evaluation script to run on Streamlit predict button click.

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone <repo_link>
cd fake_news_detector

2️⃣ Create Virtual Environment (Recommended)

```bash

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux


3️⃣ Install Dependencies
```bash

pip install -r requirements.txt
Or manually:

```bash

pip install pandas numpy scikit-learn nltk streamlit joblib matplotlib seaborn

4️⃣ Run the Pipeline Step-by-Step
Data Exploration:

```bash

python src/data_exploration.py
Data Cleaning:

```bash

python src/data_cleaning.py
Feature Engineering:

```bash

python src/feature_engineering.py
Model Training:

```bash

python src/train_model.py
Local Prediction Testing:

```bash

python src/predict.py
Run the Streamlit App:

```bash

streamlit run app/streamlit_app.py
Paste any article and check its prediction (FAKE/REAL).
Clicking the Predict button also automatically saves the test results metrics and predictions to the results/ folder.

🧩 Limitations & Next Steps
The current model has ~50% accuracy due to dataset and feature limitations.

The pipeline uses TF-IDF + Logistic Regression, which lacks semantic understanding.

For real-world deployment:

Use larger, credible datasets.

Use transformer-based embeddings (BERT, RoBERTa, etc.) for deeper context.

Perform hyperparameter tuning using tools like GridSearchCV.

Add additional features such as source credibility, article length, writing style, and metadata.

📈 Learning Outcomes
✅ End-to-end ML pipeline structuring.
✅ Modular code for data cleaning, feature engineering, training, and deployment.
✅ Practical Streamlit deployment for interactive ML testing.
✅ Integration of evaluation scripts to automate saving of test metrics and predictions.
✅ Clean GitHub-ready project structure suitable for internship portfolio.

🤝 Contribution
This project is developed as part of Tanmay Khairnar's Python & ML Internship to build real-world workflow understanding and practical skills in machine learning and deployment.

📧 Contact
For improvements, questions, or collaboration:
Tanmay Khairnar
✉️ tanmayska11@gmail.com
LinkedIn: `https://www.linkedin.com/in/tanmay-khairnar-72990314a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app`

🚀 Happy Learning & Building!










Ask ChatGPT




Tools



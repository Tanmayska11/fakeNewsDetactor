# 📰 Fake News Detector - Internship Project

## 📌 Overview

This **Fake News Detector** is a professional end-to-end machine learning pipeline built for your internship to practice **text classification, data cleaning, feature engineering, model training, and Streamlit deployment**.

It uses:
✅ **TF-IDF Vectorization**
✅ **Logistic Regression Classifier**
✅ **Streamlit App for Testing**
✅ **Modular, clean code structure** for real-developer workflow learning.

⚠️ **Disclaimer:** This project is for **learning and demonstration only**. The model currently achieves \~50% accuracy on the training dataset and is **not ready for real-world fake news detection.**

---

## 📂 Project Structure

```
fake_news_detector/
├── data/
│   ├── fake_news.csv                # Raw dataset
│   ├── clean_fake_news.csv          # Cleaned dataset
│   └── tfidf_features.npz           # Saved TF-IDF features
│
├── models/
│   ├── fake_news_model.pkl          # Trained model
│   └── tfidf_vectorizer.pkl         # Saved vectorizer
│
├── src/
│   ├── data_exploration.py          # Initial exploration
│   ├── data_cleaning.py             # Data cleaning pipeline
│   ├── feature_engineering.py       # TF-IDF vectorization
│   ├── train_model.py               # Model training and evaluation
│   ├── predict.py                   # Local prediction utility
|   └── save_test_results.py         #  Script to evaluate and save test results
│
├── app/
│   └── streamlit_app.py             # Streamlit app for testing
│
└── README.md                        # Project documentation
```

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

```
git clone https://github.com/Tanmayska11/fakeNewsDetactor.git
cd fake_news_detector
```

### 2️⃣ Create Virtual Environment (Recommended)

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

or manually:

```
pip install pandas numpy scikit-learn nltk streamlit joblib matplotlib seaborn
```

### 4️⃣ Run the Pipeline Step-by-Step

✅ **Data Exploration:**

```
python src/data_exploration.py
```

✅ **Data Cleaning:**

```
python src/data_cleaning.py
```

✅ **Feature Engineering:**

```
python src/feature_engineering.py
```

✅ **Model Training:**

```
python src/train_model.py
```

✅ **Local Prediction Testing:**

```
python src/predict.py
```

✅ **Run the Streamlit App:**

```
streamlit run app/streamlit_app.py
```

Paste any article and check its prediction (**FAKE/REAL**).

---

## 🧩 Limitations & Next Steps

* The current model has **\~50% accuracy** due to dataset and feature limitations.
* The pipeline uses **TF-IDF + Logistic Regression**, which lacks semantic understanding.
* For real-world deployment:

  * Use **larger, credible datasets**.
  * Use **BERT, RoBERTa, or embeddings** for deeper context.
  * Hyperparameter tuning with `GridSearchCV`.
  * Add additional features (source credibility, length, etc.).

---

## 📈 Learning Outcomes

✅ End-to-end ML pipeline structuring.
✅ Modular code for data cleaning, feature engineering, training, and deployment.
✅ Practical Streamlit deployment for interactive ML testing.
✅ Clean GitHub-ready project for your **internship portfolio.**

---

## 🤝 Contribution

This project is developed as part of **Tanmay's Python & ML Internship** for building real-world workflow understanding.

---

## 📧 Contact

For improvements or discussions:
**Tanmay Khairnar**
`tanmayska11@gmail.com`
LinkedIn: `https://www.linkedin.com/in/tanmay-khairnar-72990314a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app`

---

🚀 **Happy Learning & Building!**

---
## Application link

`https://fakenewsdetactor-dbgsqhi8enzk4fo5maadqg.streamlit.app/`

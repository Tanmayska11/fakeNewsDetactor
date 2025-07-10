# streamlit_app.py

import streamlit as st
import joblib
from src.save_test_results import save_test_results

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_fake_news(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return 'FAKE' if prediction == 0 else 'REAL'

def main():
    st.title("📰 Fake News Detector")
    st.write("Enter a news article below to check if it is **FAKE** or **REAL**.")

    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    text_area = st.text_area(
        "Paste your news article here:",
        value=st.session_state.text_input,
        height=300
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Predict"):
            if text_area.strip() == "":
                st.warning("⚠️ Please enter text before predicting.")
            else:
                result = predict_fake_news(text_area)
                if result == 'FAKE':
                    st.error(f"🚩 The article is predicted as: **{result}**")
                else:
                    st.success(f"✅ The article is predicted as: **{result}**")

            save_test_results()
            st.info("📁 Test results saved automatically.")   

    with col2:
        if st.button("Clear Text"):
            st.warning("⚠️ Please manually clear the textbox before entering the next article.")


if __name__ == "__main__":
    main()

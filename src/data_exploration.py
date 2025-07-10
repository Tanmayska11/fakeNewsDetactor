# data_exploration.py

import pandas as pd
import numpy as np

# Load dataset safely
try:
    df = pd.read_csv('data/fake_news_dataset.csv', encoding='utf-8')
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# 1️⃣ Basic Shape and Structure
print(f"\nShape of the dataset: {df.shape}")
print(f"\nColumns in the dataset: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# 2️⃣ Preview data
print("\nFirst 5 rows:")
print(df.head())

# 3️⃣ Missing and Empty Values
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nPercentage of missing values per column:")
print(df.isnull().mean() * 100)

print("\nEmpty string values per column:")
print((df == '').sum())

# 4️⃣ Duplicate Rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# 5️⃣ Unique Values per Column
print("\nUnique values per column:")
print(df.nunique())

# 6️⃣ Label Distribution (to check for imbalance)
if 'label' in df.columns:
    print("\nLabel distribution:")
    print(df['label'].value_counts(dropna=False))

# 7️⃣ Date Parsing Check
if 'date' in df.columns:
    df['date_converted'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"\nUnparsed dates (NaT) after conversion: {df['date_converted'].isnull().sum()}")
    print(f"\nDate range: {df['date_converted'].min()} to {df['date_converted'].max()}")

# 8️⃣ Text Length Statistics
if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).apply(len)
    print("\nText length statistics:")
    print(df['text_length'].describe())

# 9️⃣ Sample Text Content for Visual Inspection
print("\nSample text content:")
print(df['text'].dropna().sample(3, random_state=42).values)

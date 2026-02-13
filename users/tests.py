from django.test import TestCase

# Create your tests here.
# =========================================
# HOUSE RENT PREDICTION - PREPROCESSING
# Base Paper Aligned
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------
file_path = r"C:\Users\reddy\OneDrive\Desktop\house rent prediction\code\house_rent_prediction\media\House_Rent_Dataset.csv"

df = pd.read_csv(file_path)

print("Initial Shape:", df.shape)
print(df.head())

# -----------------------------------------
# 2. RENAME COLUMNS (BASE PAPER STANDARD)
# -----------------------------------------
df = df.rename(columns={
    "Rent": "price",
    "Size": "area",
    "BHK": "bedroom",
    "Bathroom": "bathroom",
    "Area Locality": "locality",
    "Furnishing Status": "furnish_type",
    "Point of Contact": "seller_type",
    "Area Type": "property_type"
})

# -----------------------------------------
# 3. DROP IRRELEVANT COLUMNS
# -----------------------------------------
df.drop(columns=["Posted On"], inplace=True)

print("\nAfter column cleanup:", df.shape)

# -----------------------------------------
# 4. HANDLE MISSING VALUES (SAFETY STEP)
# -----------------------------------------
# (Dataset usually has no missing values, but this is IEEE-safe)

df.dropna(inplace=True)

# -----------------------------------------
# 5. FEATURE ENGINEERING (FROM BASE PAPER)
# -----------------------------------------

# Price per square foot
df["price_per_sqft"] = df["price"] / df["area"]

# Bedroom to Bathroom ratio
df["bed_bath_ratio"] = df["bedroom"] / df["bathroom"]

# -----------------------------------------
# 6. HANDLE SKEWNESS (VERY IMPORTANT)
# -----------------------------------------
# Log transform target variable
df["price_log"] = np.log1p(df["price"])

# -----------------------------------------
# 7. CREATE BALANCED RENT CATEGORIES
# (Used ONLY for stratified splitting)
# -----------------------------------------
df["rent_category"] = pd.qcut(
    df["price_log"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

print("\nRent Category Distribution:")
print(df["rent_category"].value_counts())

# -----------------------------------------
# 8. SEPARATE FEATURES & TARGET
# -----------------------------------------
X = df.drop(columns=["price", "price_log", "rent_category"])
y = df["price_log"]

# -----------------------------------------
# 9. ENCODE CATEGORICAL VARIABLES
# -----------------------------------------
X = pd.get_dummies(X, drop_first=True)

print("\nTotal Features After Encoding:", X.shape[1])

# -----------------------------------------
# 10. STRATIFIED TRAIN-TEST SPLIT (4:1)
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["rent_category"]
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# -----------------------------------------
# 11. FEATURE SCALING
# -----------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 12. FINAL OUTPUT
# -----------------------------------------
print("\nPreprocessing Completed Successfully ✔")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape :", X_test_scaled.shape)
print("y_train shape       :", y_train.shape)
print("y_test shape        :", y_test.shape)
# ==========================================================
# HOUSE RENT PREDICTION - COMPLETE PREPROCESSING SCRIPT
# Base Paper Aligned | Balanced Dataset Creation
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# 1. FILE PATHS
# ----------------------------------------------------------
INPUT_FILE_PATH = r"C:\Users\reddy\OneDrive\Desktop\house rent prediction\code\house_rent_prediction\media\House_Rent_Dataset.csv"

BALANCED_DATASET_PATH = r"C:\Users\reddy\OneDrive\Desktop\house rent prediction\code\house_rent_prediction\media\House_Rent_Balanced_Dataset.csv"

ENCODED_DATASET_PATH = r"C:\Users\reddy\OneDrive\Desktop\house rent prediction\code\house_rent_prediction\media\House_Rent_Encoded_Dataset.csv"

# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv(INPUT_FILE_PATH)

print("Initial Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------
# 3. RENAME COLUMNS (BASE PAPER STANDARD)
# ----------------------------------------------------------
df = df.rename(columns={
    "Rent": "price",
    "Size": "area",
    "BHK": "bedroom",
    "Bathroom": "bathroom",
    "Area Locality": "locality",
    "Furnishing Status": "furnish_type",
    "Point of Contact": "seller_type",
    "Area Type": "property_type"
})

# ----------------------------------------------------------
# 4. DROP IRRELEVANT COLUMNS
# ----------------------------------------------------------
if "Posted On" in df.columns:
    df.drop(columns=["Posted On"], inplace=True)

print("\nAfter column cleanup:", df.shape)

# ----------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ----------------------------------------------------------
df.dropna(inplace=True)

# ----------------------------------------------------------
# 6. FEATURE ENGINEERING (BASE PAPER)
# ----------------------------------------------------------
df["price_per_sqft"] = df["price"] / df["area"]
df["bed_bath_ratio"] = df["bedroom"] / df["bathroom"]

# ----------------------------------------------------------
# 7. HANDLE SKEWNESS (LOG TRANSFORMATION)
# ----------------------------------------------------------
df["price_log"] = np.log1p(df["price"])

# ----------------------------------------------------------
# 8. CREATE BALANCED RENT CATEGORIES
# ----------------------------------------------------------
df["rent_category"] = pd.qcut(
    df["price_log"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

print("\nRent Category Distribution:")
print(df["rent_category"].value_counts())

# ----------------------------------------------------------
# 9. SAVE BALANCED DATASET (FOR PAPER & REUSE)
# ----------------------------------------------------------
df.to_csv(BALANCED_DATASET_PATH, index=False)

print("\nBalanced dataset saved successfully ✅")
print("Saved at:", BALANCED_DATASET_PATH)

# ----------------------------------------------------------
# 10. PREPARE MODEL-READY DATA
# ----------------------------------------------------------
X = df.drop(columns=["price", "price_log", "rent_category"])
y = df["price_log"]

# ----------------------------------------------------------
# 11. ONE-HOT ENCODING
# ----------------------------------------------------------
X_encoded = pd.get_dummies(X, drop_first=True)

print("\nTotal Features After Encoding:", X_encoded.shape[1])

# ----------------------------------------------------------
# 12. STRATIFIED TRAIN-TEST SPLIT (4:1)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["rent_category"]
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ----------------------------------------------------------
# 13. FEATURE SCALING
# ----------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 14. SAVE ENCODED DATASET (OPTIONAL)
# ----------------------------------------------------------
encoded_dataset = X_encoded.copy()
encoded_dataset["price_log"] = y.values

encoded_dataset.to_csv(ENCODED_DATASET_PATH, index=False)

print("\nEncoded dataset saved successfully ✅")
print("Saved at:", ENCODED_DATASET_PATH)

# ----------------------------------------------------------
# 15. FINAL CONFIRMATION
# ----------------------------------------------------------
print("\n==============================")
print("PREPROCESSING COMPLETED ✔")
print("==============================")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape :", X_test_scaled.shape)
print("y_train shape       :", y_train.shape)
print("y_test shape        :", y_test.shape)

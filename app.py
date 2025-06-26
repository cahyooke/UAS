import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Load data preprocessed ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_cancer_preprocessed.csv")
    return df

df = load_data()

# Pisahkan fitur dan target
X = df.drop(columns="Class")
y = df["Class"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (for training models)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Models ---
model_dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
model_nb = GaussianNB().fit(X_train, y_train)
model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

# --- UI ---
st.title(" Prediksi Kanker Payudara")
st.markdown("Masukkan nilai fitur selanjutnya untuk memprediksi apakah kanker **benign (2)** atau **malignant (4)**.")

# --- Input Fitur Manual ---
feature_inputs = {}
for col in X.columns:
    feature_inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# Convert ke array dan scaling
input_df = pd.DataFrame([feature_inputs])
input_scaled = scaler.transform(input_df)

# --- Pilih Model ---
model_choice = st.selectbox(" Pilih Model Klasifikasi", ['Decision Tree', 'Naive Bayes', 'KNN'])

# --- Prediksi ---
if st.button("Prediksi"):
    if model_choice == 'Decision Tree':
        pred = model_dt.predict(input_scaled)[0]
    elif model_choice == 'Naive Bayes':
        pred = model_nb.predict(input_scaled)[0]
    elif model_choice == 'KNN':
        pred = model_knn.predict(input_scaled)[0]
    
    hasil = "🟢 Jinak (Benign)" if pred == 2 else "🔴 Ganas (Malignant)"
    st.subheader(f"Hasil Prediksi: {hasil}")
    st.write(f"Kode Prediksi Model: **{pred}**")

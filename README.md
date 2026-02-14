# 📊 Customer Segmentation & Churn Prediction Dashboard

An end-to-end data science project that segments customers, analyzes churn behavior, predicts churn risk, and delivers business-ready insights through an interactive dashboard.

---

## Project Overview

Customer churn directly impacts revenue and long-term growth.  

This project demonstrates how data science can help businesses:

- Understand customer behavior  
- Identify high-risk, high-value customers  
- Predict churn probability  
- Support targeted retention strategies  

The solution combines unsupervised learning, supervised modeling, interpretability, and deployment into a complete decision-support system.

---

## Key Objectives

- Segment customers based on behavioral patterns  
- Analyze churn distribution across customer segments  
- Predict churn probability at the individual customer level  
- Identify key drivers of churn  
- Provide an interactive dashboard for business users  

---

## 📂 Project Structure

```bash

Customer-Churn-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── churn_model.pkl
│   └── model_columns.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_churn_analysis.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_interpretation.ipynb
│
├── app.py
├── requirements.txt
└── README.md

```

---

## Methodology

### 1 - Exploratory Data Analysis (EDA)

- Examined customer demographics, service subscriptions, billing information  
- Analyzed churn patterns across tenure, payment methods, and contract types  
- Identified potential risk indicators  

---

### 2 - Data Cleaning & Feature Engineering

- Converted data types (e.g., `TotalCharges`)  
- Encoded categorical variables  
- Removed non-predictive identifiers (`customerID` from model features)  
- Prepared dataset for modeling  

---

### 3 - Customer Segmentation (Unsupervised Learning)

- Applied K-Means clustering  
- Grouped customers by behavioral patterns  

Identified:

- High-value loyal customers  
- Price-sensitive customers  
- Short-tenure high-risk customers  

- Analyzed churn rates per segment  

---

### 4 - Churn Prediction Model (Supervised Learning)

- Trained a Logistic Regression model  
- Generated churn probability scores  

Evaluated performance using:

- Accuracy  
- Precision / Recall  
- Confusion Matrix  

Focused on interpretability over complexity.

---

### 5 - Model Interpretability

- Extracted and analyzed feature coefficients  
- Identified top positive churn drivers  
- Enabled per-customer risk explanations  

---

### 6 - Deployment (Streamlit Dashboard)

The interactive dashboard allows users to:

- Filter customers by churn probability  
- Analyze churn rate by segment  
- Inspect individual customers  
- View top churn drivers  
- Simulate retention prioritization  

---

## 🖥️ Live Demo

🔗 https://customer-churn-risk-dashboard.streamlit.app/

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Joblib  
- Docker (optional containerization)  

---

## 📈 Business Impact

This project demonstrates how businesses can:

- Prioritize retention resources toward high-risk, high-value customers  
- Understand structural churn drivers (e.g., contract type, tenure)  
- Move from reactive churn analysis to proactive risk management  
- Use interpretable ML models for decision-making  

---

## Why This Project Matters?

Rather than focusing on model accuracy, this project emphasizes:

- End-to-end data science workflow  
- Business interpretability  
- Deployment readiness  
- Practical decision-support systems  

It simulates how churn analytics would operate inside a real organization.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
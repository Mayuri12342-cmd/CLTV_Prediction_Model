# 💰 Customer Lifetime Value (CLTV) Prediction App

An end-to-end Machine Learning project that predicts Customer Lifetime Value (CLTV) using a Random Forest model and provides an interactive Streamlit dashboard for business insights.

---

## 🚀 Live Features

- 🔮 CLTV prediction using ML (Random Forest)
- 📊 Revenue distribution analysis
- 🧠 RFM segmentation (Champions, Loyal, Potential, At Risk)
- 👥 Customer clustering using KMeans
- 📈 Interactive Plotly visualizations
- 📋 Top customer analytics table
- 💡 Business-ready insights dashboard

---

## 🧠 Problem Statement

Businesses want to identify **high-value customers** and optimize marketing strategies.

This project predicts:
- How much revenue a customer will generate (CLTV)
- Customer segments based on behavior
- Hidden patterns using clustering

---

## 📂 Dataset Used

- Olist E-commerce Dataset (Brazil)
  - Customers
  - Orders
  - Order Items
  - Payments

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn (Random Forest, KMeans)
- Streamlit (UI)
- Plotly (Visualization)
- Joblib (Model saving)

---

## 🏗️ Project Workflow

1. Data Cleaning & Merging
2. Feature Engineering
   - Monetary value
   - Frequency
   - Recency
3. Model Training (Random Forest Regressor)
4. RFM Segmentation
5. Customer Clustering (KMeans)
6. Streamlit Dashboard

---

## 🔮 Features Used for Prediction

- Total Spend (monetary_sum)
- Average Order Value (monetary_avg)
- Number of Purchases (frequency)
- Recency (days since last purchase)

---

## 📊 Dashboard Includes

- CLTV Prediction Panel (Sidebar)
- Revenue Distribution (Histogram)
- RFM Segments (Pie Chart)
- Frequency vs Monetary (Scatter Plot)
- Customer Clusters (KMeans visualization)
- Top Customers Table

---

## 🖥️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cltv-project.git
cd cltv-project

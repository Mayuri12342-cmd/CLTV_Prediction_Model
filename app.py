import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.cluster import KMeans

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="CLTV Dashboard", layout="wide")
st.title("💰 CLTV Analytics Dashboard (Pro Version)")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("rf_clv_model.pkl")

# -------------------------
# DATA BUILD
# -------------------------
customers = pd.read_csv("dataset/olist_customers_dataset.csv")
orders = pd.read_csv("dataset/olist_orders_dataset.csv", parse_dates=['order_purchase_timestamp'])
order_items = pd.read_csv("dataset/olist_order_items_dataset.csv")
payments = pd.read_csv("dataset/olist_order_payments_dataset.csv")

payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
prices_agg = order_items.groupby('order_id')['price'].sum().reset_index()

orders_full = orders.merge(payments_agg, on='order_id', how='left')
orders_full = orders_full.merge(prices_agg, on='order_id', how='left')

orders_full['order_value'] = orders_full['payment_value'].fillna(orders_full['price']).fillna(0)

orders_full = orders_full[orders_full['order_status'] != 'canceled']
orders_full = orders_full.merge(customers, on='customer_id', how='left')

customer_agg = orders_full.groupby('customer_id').agg({
    'order_value': ['sum', 'mean', 'count'],
    'order_purchase_timestamp': ['min', 'max']
})

customer_agg.columns = ['monetary_sum','monetary_avg','frequency','first_order','last_order']
customer_agg = customer_agg.reset_index()

customer_agg['recency_days'] = (customer_agg['last_order'] - customer_agg['first_order']).dt.days

df = customer_agg.copy()

# -------------------------
# SIDEBAR PREDICTION
# -------------------------
st.sidebar.header("🔮 CLTV Prediction")

monetary_sum = st.sidebar.number_input("Total Spend", 0.0)
monetary_avg = st.sidebar.number_input("Avg Order Value", 0.0)
frequency = st.sidebar.number_input("Frequency", 0)
recency_days = st.sidebar.number_input("Recency (Days)", 0)

if st.sidebar.button("Predict CLTV"):

    input_data = np.array([[monetary_sum,
                            monetary_avg,
                            frequency,
                            recency_days]])

    pred = model.predict(input_data)[0]

    st.sidebar.success(f"Predicted CLTV: ₹{round(pred,2)}")

    if pred < 20000:
        st.sidebar.info("Low Value Customer")
    elif pred < 60000:
        st.sidebar.info("Medium Value Customer")
    else:
        st.sidebar.info("High Value Customer")

# -------------------------
# SAFE RFM
# -------------------------
df["R_Score"] = pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4,3,2,1])
df["F_Score"] = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1,2,3,4])
df["M_Score"] = pd.qcut(df["monetary_sum"].rank(method="first"), 4, labels=[1,2,3,4])

df["RFM_Score"] = df["R_Score"].astype(int) + df["F_Score"].astype(int) + df["M_Score"].astype(int)

def segment(x):
    if x >= 10:
        return "Champions"
    elif x >= 7:
        return "Loyal"
    elif x >= 5:
        return "Potential"
    else:
        return "At Risk"

df["Segment"] = df["RFM_Score"].apply(segment)

# -------------------------
# CLUSTERING
# -------------------------
features = df[["monetary_sum", "frequency", "recency_days"]]
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(features)

# -------------------------
# METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg CLV", round(df["monetary_sum"].mean(), 2))
col3.metric("Champions", len(df[df["Segment"] == "Champions"]))

st.markdown("---")

# -------------------------
# CHART 1: REVENUE
# -------------------------
st.subheader("💰 Revenue Distribution")
fig1 = px.histogram(df, x="monetary_sum", nbins=40)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# CHART 2: RFM
# -------------------------
st.subheader("🧠 RFM Segments")
fig2 = px.pie(df, names="Segment")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# CHART 3: FREQUENCY VS MONETARY
# -------------------------
st.subheader("📊 Frequency vs Monetary")
fig3 = px.scatter(df,
                  x="frequency",
                  y="monetary_sum",
                  color="Segment",
                  size="monetary_sum")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# CHART 4: CLUSTERS
# -------------------------
st.subheader("👥 Customer Clusters")
fig4 = px.scatter(df,
                  x="frequency",
                  y="monetary_sum",
                  color=df["Cluster"].astype(str))
st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# TABLE 1: TOP CUSTOMERS
# -------------------------
st.subheader("🏆 Top Customers Table")

top_df = df.sort_values("monetary_sum", ascending=False).head(15)

st.dataframe(top_df[[
    "customer_id",
    "monetary_sum",
    "monetary_avg",
    "frequency",
    "recency_days",
    "Segment",
    "Cluster"
]], use_container_width=True)
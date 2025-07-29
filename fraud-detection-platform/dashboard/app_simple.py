import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load transaction data from CSV files"""
    try:
        transactions = pd.read_csv('../data/raw/transactions.csv')
        customers = pd.read_csv('../data/raw/customers.csv')
        merchants = pd.read_csv('../data/raw/merchants.csv')
        
        # Parse timestamp
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['transaction_date'] = transactions['timestamp'].dt.date
        transactions['hour_of_day'] = transactions['timestamp'].dt.hour
        transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
        
        # Merge data
        df = transactions.merge(customers, on='customer_id', how='left')
        df = df.merge(merchants, on='merchant_id', how='left')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def predict_fraud(transaction_data):
    """Simple rule-based fraud prediction for demo"""
    score = 0
    
    # High amount transactions
    if transaction_data['amount'] > 1000:
        score += 0.3
    
    # Night transactions
    if transaction_data['hour_of_day'] in [0, 1, 2, 3, 4, 5, 22, 23]:
        score += 0.2
    
    # High risk merchant
    if transaction_data.get('risk_score', 0) > 0.7:
        score += 0.25
    
    # Very small amounts (testing)
    if transaction_data['amount'] < 1:
        score += 0.25
    
    return min(score, 1.0)

def main():
    st.title("🔒 Real-Time Fraud Detection Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Overview", 
        "Transaction Monitoring", 
        "Analytics", 
        "Real-time Prediction"
    ])
    
    if page == "Overview":
        show_overview()
    elif page == "Transaction Monitoring":
        show_transaction_monitoring()
    elif page == "Analytics":
        show_analytics()
    elif page == "Real-time Prediction":
        show_prediction_interface()

def show_overview():
    st.header("Fraud Detection Overview")
    
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please run the data generation script first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        fraud_rate = df['is_fraud'].mean() * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        total_amount = df['amount'].sum()
        st.metric("Total Transaction Value", f"${total_amount:,.2f}")
    
    # Fraud detection over time
    st.subheader("Fraud Detection Trends")
    
    daily_stats = df.groupby('transaction_date').agg({
        'is_fraud': ['count', 'sum'],
        'amount': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'total_transactions', 'fraud_transactions', 'total_amount']
    daily_stats['fraud_rate'] = daily_stats['fraud_transactions'] / daily_stats['total_transactions'] * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Fraud Rate (%)', 'Daily Transaction Volume'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['fraud_rate'], 
                  name='Fraud Rate %', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=daily_stats['date'], y=daily_stats['total_transactions'], 
               name='Total Transactions', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def show_transaction_monitoring():
    st.header("Transaction Monitoring")
    
    df = load_data()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraud_filter = st.selectbox("Transaction Type", ["All", "Fraudulent Only", "Legitimate Only"])
    
    with col2:
        category_filter = st.selectbox("Merchant Category", ["All"] + list(df['category'].unique()))
    
    with col3:
        amount_filter = st.slider("Min Amount", 0, int(df['amount'].max()), 0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if fraud_filter == "Fraudulent Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == True]
    elif fraud_filter == "Legitimate Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == False]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    filtered_df = filtered_df[filtered_df['amount'] >= amount_filter]
    
    # Recent transactions table
    st.subheader("Recent Transactions")
    
    display_df = filtered_df[['transaction_id', 'amount', 'category', 
                             'payment_method', 'timestamp', 'is_fraud']].head(20)
    
    # Color code the dataframe
    def color_rows(row):
        if row['is_fraud']:
            return ['background-color: #ffebee'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(display_df.style.apply(color_rows, axis=1), use_container_width=True)
    
    # Amount distribution
    st.subheader("Transaction Amount Distribution")
    
    fig = px.histogram(filtered_df, x='amount', color='is_fraud', 
                      nbins=30, title='Amount Distribution by Fraud Status')
    st.plotly_chart(fig, use_container_width=True)

def show_analytics():
    st.header("Fraud Analytics")
    
    df = load_data()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by merchant category
        fraud_by_category = df.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_category.columns = ['category', 'total_transactions', 'fraud_count', 'fraud_rate']
        fraud_by_category['fraud_rate'] = fraud_by_category['fraud_rate'] * 100
        
        fig = px.bar(fraud_by_category, x='category', y='fraud_rate',
                    title='Fraud Rate by Merchant Category (%)')
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud by hour of day
        fraud_by_hour = df.groupby('hour_of_day')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_hour.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']
        fraud_by_hour['fraud_rate'] = fraud_by_hour['fraud_rate'] * 100
        
        fig = px.line(fraud_by_hour, x='hour', y='fraud_rate',
                     title='Fraud Rate by Hour of Day (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction amount analysis
    st.subheader("Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='is_fraud', y='amount', 
                    title='Transaction Amount Distribution by Fraud Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer age vs fraud
        fig = px.histogram(df, x='age', color='is_fraud', 
                         title='Customer Age Distribution by Fraud Status',
                         nbins=20)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_interface():
    st.header("Real-time Fraud Prediction")
    
    st.write("Enter transaction details to get real-time fraud prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
        merchant_category = st.selectbox("Merchant Category", 
                                       ['grocery', 'restaurant', 'gas_station', 'retail', 'online',
                                        'entertainment', 'travel', 'healthcare', 'utilities', 'other'])
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Sunday)", 0, 6, 1)
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    
    with col2:
        payment_method = st.selectbox("Payment Method", ['credit_card', 'debit_card', 'mobile_payment', 'bank_transfer'])
        merchant_risk_score = st.slider("Merchant Risk Score", 0.0, 1.0, 0.3, 0.01)
        is_weekend = st.checkbox("Weekend Transaction")
        customer_id = st.text_input("Customer ID", "CUST_001234")
    
    if st.button("Predict Fraud Risk", type="primary"):
        transaction_data = {
            'amount': amount,
            'merchant_category': merchant_category,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'customer_age': customer_age,
            'payment_method': payment_method,
            'risk_score': merchant_risk_score,
            'is_weekend': is_weekend,
            'customer_id': customer_id
        }
        
        fraud_probability = predict_fraud(transaction_data)
        is_fraud_predicted = fraud_probability > 0.5
        
        if fraud_probability < 0.3:
            risk_level = "Low"
            risk_color = "🟢"
        elif fraud_probability < 0.7:
            risk_level = "Medium"  
            risk_color = "🟡"
        else:
            risk_level = "High"
            risk_color = "🔴"
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Probability", f"{fraud_probability:.3f}")
        
        with col2:
            status = "🚨 FRAUD" if is_fraud_predicted else "✅ LEGITIMATE"
            st.metric("Predicted Status", status)
        
        with col3:
            st.metric("Risk Level", f"{risk_color} {risk_level}")
        
        # Recommendation
        if is_fraud_predicted:
            st.error("⚠️ This transaction is flagged as potentially fraudulent. Recommend manual review.")
        else:
            st.success("✅ This transaction appears legitimate.")
        
        # Show transaction details
        with st.expander("Transaction Details"):
            st.json(transaction_data)

if __name__ == "__main__":
    main()
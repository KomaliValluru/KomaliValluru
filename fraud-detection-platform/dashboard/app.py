import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ml.predict import FraudPredictor

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide"
)

@st.cache_data
def load_data():
    conn = duckdb.connect('../data/fraud_detection.duckdb')
    
    query = """
    SELECT 
        transaction_id,
        customer_id,
        merchant_id,
        amount,
        transaction_timestamp,
        transaction_date,
        payment_method,
        merchant_category,
        is_fraud,
        customer_age,
        hour_of_day,
        day_of_week,
        calculated_risk_score,
        merchant_risk_score
    FROM mart_fraud_detection
    ORDER BY transaction_timestamp DESC
    LIMIT 10000
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    return df

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
    
    try:
        df = load_data()
        
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
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
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
        
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['total_amount'], 
                      name='Total Amount', line=dict(color='green')),
            row=2, col=1, secondary_y=True
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure you have run the data generation and dbt transformations first.")

def show_transaction_monitoring():
    st.header("Real-time Transaction Monitoring")
    
    try:
        df = load_data()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox("Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])
        
        with col2:
            category_filter = st.selectbox("Merchant Category", ["All"] + list(df['merchant_category'].unique()))
        
        with col3:
            amount_filter = st.slider("Min Amount", 0, int(df['amount'].max()), 0)
        
        # Apply filters
        filtered_df = df.copy()
        
        if risk_filter != "All":
            if risk_filter == "High Risk":
                filtered_df = filtered_df[filtered_df['calculated_risk_score'] > 0.7]
            elif risk_filter == "Medium Risk":
                filtered_df = filtered_df[(filtered_df['calculated_risk_score'] > 0.3) & (filtered_df['calculated_risk_score'] <= 0.7)]
            else:
                filtered_df = filtered_df[filtered_df['calculated_risk_score'] <= 0.3]
        
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['merchant_category'] == category_filter]
        
        filtered_df = filtered_df[filtered_df['amount'] >= amount_filter]
        
        # Recent transactions table
        st.subheader("Recent Transactions")
        
        display_df = filtered_df[['transaction_id', 'amount', 'merchant_category', 
                                 'payment_method', 'calculated_risk_score', 'is_fraud']].head(20)
        
        # Color code based on fraud status
        def highlight_fraud(row):
            if row['is_fraud']:
                return ['background-color: #ffebee'] * len(row)
            elif row['calculated_risk_score'] > 0.7:
                return ['background-color: #fff3e0'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(display_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
        
        # Risk distribution
        st.subheader("Risk Score Distribution")
        
        fig = px.histogram(filtered_df, x='calculated_risk_score', 
                          color='is_fraud', nbins=20,
                          title='Distribution of Risk Scores')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")

def show_analytics():
    st.header("Fraud Analytics")
    
    try:
        df = load_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by merchant category
            fraud_by_category = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            fraud_by_category.columns = ['category', 'total_transactions', 'fraud_count', 'fraud_rate']
            fraud_by_category['fraud_rate'] = fraud_by_category['fraud_rate'] * 100
            
            fig = px.bar(fraud_by_category, x='category', y='fraud_rate',
                        title='Fraud Rate by Merchant Category (%)')
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
            fig = px.histogram(df, x='customer_age', color='is_fraud', 
                             title='Customer Age Distribution by Fraud Status',
                             nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

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
        transaction_frequency = st.number_input("Customer Transaction Frequency", min_value=1, value=50)
        amount_percentile = st.number_input("Amount Z-Score", value=0.0, step=0.1)
        is_night_transaction = st.checkbox("Night Transaction (10PM - 6AM)")
        is_weekend_transaction = st.checkbox("Weekend Transaction")
        is_amount_outlier = st.checkbox("Amount Outlier")
        merchant_risk_score = st.slider("Merchant Risk Score", 0.0, 1.0, 0.3, 0.01)
    
    if st.button("Predict Fraud Risk"):
        try:
            predictor = FraudPredictor()
            
            transaction_data = {
                'amount': amount,
                'merchant_category': merchant_category,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'customer_age': customer_age,
                'transaction_frequency': transaction_frequency,
                'amount_percentile': amount_percentile,
                'is_night_transaction': 1 if is_night_transaction else 0,
                'is_weekend_transaction': 1 if is_weekend_transaction else 0,
                'is_amount_outlier': 1 if is_amount_outlier else 0,
                'merchant_risk_score': merchant_risk_score
            }
            
            result = predictor.predict_single_transaction(transaction_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.3f}")
            
            with col2:
                st.metric("Predicted Status", "FRAUD" if result['is_fraud_predicted'] else "LEGITIMATE")
            
            with col3:
                risk_color = "🔴" if result['risk_level'] == 'High' else "🟡" if result['risk_level'] == 'Medium' else "🟢"
                st.metric("Risk Level", f"{risk_color} {result['risk_level']}")
            
            # Recommendation
            if result['is_fraud_predicted']:
                st.error("⚠️ This transaction is flagged as potentially fraudulent. Recommend manual review.")
            else:
                st.success("✅ This transaction appears legitimate.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please make sure the model has been trained first by running 'python src/ml/train_model.py'")

if __name__ == "__main__":
    main()
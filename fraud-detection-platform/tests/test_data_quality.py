import pytest
import pandas as pd
import duckdb
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataQuality:
    
    @pytest.fixture
    def db_connection(self):
        return duckdb.connect('../data/fraud_detection.duckdb')
    
    def test_transaction_data_completeness(self, db_connection):
        """Test that transaction data has no missing critical fields"""
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(transaction_id) as non_null_transaction_id,
            COUNT(customer_id) as non_null_customer_id,
            COUNT(merchant_id) as non_null_merchant_id,
            COUNT(amount) as non_null_amount
        FROM stg_transactions
        """
        
        result = db_connection.execute(query).df()
        
        assert result['total_records'].iloc[0] > 0, "No transaction data found"
        assert result['non_null_transaction_id'].iloc[0] == result['total_records'].iloc[0], "Missing transaction IDs"
        assert result['non_null_customer_id'].iloc[0] == result['total_records'].iloc[0], "Missing customer IDs"
        assert result['non_null_merchant_id'].iloc[0] == result['total_records'].iloc[0], "Missing merchant IDs"
        assert result['non_null_amount'].iloc[0] == result['total_records'].iloc[0], "Missing transaction amounts"
    
    def test_transaction_amount_validity(self, db_connection):
        """Test that transaction amounts are within reasonable ranges"""
        query = """
        SELECT 
            MIN(amount) as min_amount,
            MAX(amount) as max_amount,
            AVG(amount) as avg_amount,
            COUNT(CASE WHEN amount < 0 THEN 1 END) as negative_amounts,
            COUNT(CASE WHEN amount > 100000 THEN 1 END) as excessive_amounts
        FROM stg_transactions
        """
        
        result = db_connection.execute(query).df()
        
        assert result['negative_amounts'].iloc[0] == 0, "Found negative transaction amounts"
        assert result['excessive_amounts'].iloc[0] < result.iloc[0].name * 0.01, "Too many excessive amounts"  # Less than 1%
        assert result['min_amount'].iloc[0] >= 0, "Minimum amount should be non-negative"
    
    def test_fraud_rate_reasonableness(self, db_connection):
        """Test that fraud rate is within expected bounds"""
        query = """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_transactions,
            AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) as fraud_rate
        FROM stg_transactions
        """
        
        result = db_connection.execute(query).df()
        fraud_rate = result['fraud_rate'].iloc[0]
        
        assert 0.005 <= fraud_rate <= 0.1, f"Fraud rate {fraud_rate:.3f} is outside expected range (0.5% - 10%)"
    
    def test_customer_age_validity(self, db_connection):
        """Test that customer ages are reasonable"""
        query = """
        SELECT 
            MIN(customer_age) as min_age,
            MAX(customer_age) as max_age,
            COUNT(CASE WHEN customer_age < 18 THEN 1 END) as underage_customers,
            COUNT(CASE WHEN customer_age > 100 THEN 1 END) as overage_customers
        FROM stg_customers
        """
        
        result = db_connection.execute(query).df()
        
        assert result['underage_customers'].iloc[0] == 0, "Found underage customers"
        assert result['overage_customers'].iloc[0] == 0, "Found customers over 100 years old"
        assert 18 <= result['min_age'].iloc[0] <= 100, "Minimum age out of range"
        assert 18 <= result['max_age'].iloc[0] <= 100, "Maximum age out of range"
    
    def test_merchant_risk_score_range(self, db_connection):
        """Test that merchant risk scores are between 0 and 1"""
        query = """
        SELECT 
            MIN(merchant_risk_score) as min_risk,
            MAX(merchant_risk_score) as max_risk,
            COUNT(CASE WHEN merchant_risk_score < 0 OR merchant_risk_score > 1 THEN 1 END) as invalid_scores
        FROM stg_merchants
        """
        
        result = db_connection.execute(query).df()
        
        assert result['invalid_scores'].iloc[0] == 0, "Found invalid merchant risk scores"
        assert 0 <= result['min_risk'].iloc[0] <= 1, "Minimum risk score out of range"
        assert 0 <= result['max_risk'].iloc[0] <= 1, "Maximum risk score out of range"
    
    def test_transaction_timestamp_validity(self, db_connection):
        """Test that transaction timestamps are reasonable"""
        query = """
        SELECT 
            MIN(transaction_timestamp) as min_timestamp,
            MAX(transaction_timestamp) as max_timestamp,
            COUNT(CASE WHEN transaction_timestamp IS NULL THEN 1 END) as null_timestamps
        FROM stg_transactions
        """
        
        result = db_connection.execute(query).df()
        
        assert result['null_timestamps'].iloc[0] == 0, "Found null timestamps"
        
        # Check if timestamps are within the last 2 years
        min_ts = pd.to_datetime(result['min_timestamp'].iloc[0])
        max_ts = pd.to_datetime(result['max_timestamp'].iloc[0])
        now = pd.Timestamp.now()
        
        assert min_ts >= (now - pd.Timedelta(days=730)), "Timestamps too old"
        assert max_ts <= now, "Future timestamps found"

if __name__ == "__main__":
    pytest.main([__file__])
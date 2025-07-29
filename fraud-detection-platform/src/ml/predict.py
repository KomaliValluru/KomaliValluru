import pandas as pd
import numpy as np
from train_model import FraudDetectionModel
import yaml

class FraudPredictor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = FraudDetectionModel(config_path)
        self.model.load_model()
        
    def predict_single_transaction(self, transaction_data):
        """
        Predict fraud probability for a single transaction
        
        Args:
            transaction_data (dict): Dictionary containing transaction features
        
        Returns:
            float: Fraud probability (0-1)
        """
        df = pd.DataFrame([transaction_data])
        fraud_probability = self.model.predict(df)[0]
        
        return {
            'fraud_probability': fraud_probability,
            'is_fraud_predicted': fraud_probability > 0.5,
            'risk_level': self._get_risk_level(fraud_probability)
        }
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud probabilities for a batch of transactions
        
        Args:
            transactions_df (pd.DataFrame): DataFrame containing transaction features
        
        Returns:
            pd.DataFrame: Original data with fraud predictions
        """
        fraud_probabilities = self.model.predict(transactions_df)
        
        result_df = transactions_df.copy()
        result_df['fraud_probability'] = fraud_probabilities
        result_df['is_fraud_predicted'] = fraud_probabilities > 0.5
        result_df['risk_level'] = result_df['fraud_probability'].apply(self._get_risk_level)
        
        return result_df
    
    def _get_risk_level(self, probability):
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'

if __name__ == "__main__":
    predictor = FraudPredictor()
    
    # Example transaction
    sample_transaction = {
        'amount': 1500.00,
        'merchant_category': 'online',
        'hour_of_day': 2,
        'day_of_week': 1,
        'customer_age': 35,
        'transaction_frequency': 50,
        'amount_percentile': 2.5,
        'is_night_transaction': 1,
        'is_weekend_transaction': 0,
        'is_amount_outlier': 1,
        'merchant_risk_score': 0.8
    }
    
    result = predictor.predict_single_transaction(sample_transaction)
    print(f"Fraud Prediction: {result}")
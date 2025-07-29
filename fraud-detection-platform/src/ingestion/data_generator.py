import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import yaml
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class TransactionDataGenerator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.n_transactions = self.config['data_generation']['n_transactions']
        self.fraud_rate = self.config['data_generation']['fraud_rate']
        self.start_date = datetime.strptime(self.config['data_generation']['start_date'], '%Y-%m-%d')
        self.end_date = datetime.strptime(self.config['data_generation']['end_date'], '%Y-%m-%d')
        
        self.merchant_categories = [
            'grocery', 'restaurant', 'gas_station', 'retail', 'online',
            'entertainment', 'travel', 'healthcare', 'utilities', 'other'
        ]
        
        self.payment_methods = ['credit_card', 'debit_card', 'mobile_payment', 'bank_transfer']
    
    def generate_customers(self, n_customers=1000):
        customers = []
        for i in range(n_customers):
            customers.append({
                'customer_id': f'CUST_{i:06d}',
                'age': np.random.normal(40, 15),
                'account_age_days': np.random.exponential(365),
                'credit_score': np.random.normal(700, 100),
                'income': np.random.lognormal(10.5, 0.5)
            })
        return pd.DataFrame(customers)
    
    def generate_merchants(self, n_merchants=500):
        merchants = []
        for i in range(n_merchants):
            merchants.append({
                'merchant_id': f'MERCH_{i:05d}',
                'merchant_name': fake.company(),
                'category': np.random.choice(self.merchant_categories),
                'risk_score': np.random.beta(2, 5),
                'location': fake.city()
            })
        return pd.DataFrame(merchants)
    
    def generate_transactions(self, customers, merchants):
        transactions = []
        n_fraud = int(self.n_transactions * self.fraud_rate)
        
        for i in range(self.n_transactions):
            customer = customers.sample(1).iloc[0]
            merchant = merchants.sample(1).iloc[0]
            
            timestamp = fake.date_time_between(
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            is_fraud = i < n_fraud
            
            if is_fraud:
                amount = self._generate_fraud_amount(customer, merchant)
                hour = np.random.choice([2, 3, 4, 23, 0, 1], p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
            else:
                amount = self._generate_normal_amount(customer, merchant)
                hour = np.random.choice(range(24), p=self._get_hour_probabilities())
            
            timestamp = timestamp.replace(hour=hour)
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'customer_id': customer['customer_id'],
                'merchant_id': merchant['merchant_id'],
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'payment_method': np.random.choice(self.payment_methods),
                'merchant_category': merchant['category'],
                'is_fraud': is_fraud
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def _generate_normal_amount(self, customer, merchant):
        base_amount = np.random.lognormal(3, 1.5)
        
        if merchant['category'] == 'grocery':
            return max(10, np.random.normal(75, 30))
        elif merchant['category'] == 'restaurant':
            return max(5, np.random.normal(45, 20))
        elif merchant['category'] == 'gas_station':
            return max(20, np.random.normal(60, 15))
        elif merchant['category'] == 'retail':
            return max(15, np.random.normal(120, 80))
        else:
            return max(5, base_amount)
    
    def _generate_fraud_amount(self, customer, merchant):
        if np.random.random() < 0.3:
            return np.random.uniform(1000, 5000)
        elif np.random.random() < 0.5:
            return np.random.uniform(0.01, 5)
        else:
            return self._generate_normal_amount(customer, merchant) * np.random.uniform(2, 10)
    
    def _get_hour_probabilities(self):
        probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
            0.03, 0.05, 0.07, 0.06, 0.05, 0.06,  # 6-11
            0.08, 0.07, 0.06, 0.05, 0.06, 0.07,  # 12-17
            0.08, 0.07, 0.05, 0.04, 0.03, 0.02   # 18-23
        ])
        return probs / probs.sum()
    
    def generate_all_data(self):
        print("Generating customers...")
        customers = self.generate_customers()
        
        print("Generating merchants...")
        merchants = self.generate_merchants()
        
        print("Generating transactions...")
        transactions = self.generate_transactions(customers, merchants)
        
        # Save to files
        os.makedirs('data/raw', exist_ok=True)
        customers.to_csv('data/raw/customers.csv', index=False)
        merchants.to_csv('data/raw/merchants.csv', index=False)
        transactions.to_csv('data/raw/transactions.csv', index=False)
        
        print(f"Generated {len(transactions)} transactions ({transactions['is_fraud'].sum()} fraudulent)")
        print("Data saved to data/raw/")
        
        return customers, merchants, transactions

if __name__ == "__main__":
    generator = TransactionDataGenerator()
    customers, merchants, transactions = generator.generate_all_data()
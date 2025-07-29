import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import yaml

class FraudDetectionModel:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data from CSV files"""
        # Load data
        transactions = pd.read_csv('data/raw/transactions.csv')
        customers = pd.read_csv('data/raw/customers.csv')
        merchants = pd.read_csv('data/raw/merchants.csv')
        
        # Parse timestamp
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['hour_of_day'] = transactions['timestamp'].dt.hour
        transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
        
        # Merge dataframes
        df = transactions.merge(customers, on='customer_id', how='left')
        df = df.merge(merchants, on='merchant_id', how='left')
        
        # Feature engineering
        customer_stats = transactions.groupby('customer_id').agg({
            'amount': ['count', 'mean', 'std']
        }).reset_index()
        customer_stats.columns = ['customer_id', 'transaction_frequency', 'avg_amount', 'std_amount']
        customer_stats['std_amount'] = customer_stats['std_amount'].fillna(0)
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Calculate z-score for amount
        df['amount_zscore'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1e-6)
        
        # Binary features
        df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_weekend_transaction'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_amount_outlier'] = (np.abs(df['amount_zscore']) > 2).astype(int)
        df['is_high_risk_merchant'] = (df['risk_score'] > 0.7).astype(int)
        
        return df
    
    def preprocess_features(self, df, is_training=True):
        """Preprocess features for ML model"""
        df_processed = df.copy()
        
        # Select features for training
        feature_cols = [
            'amount', 'hour_of_day', 'day_of_week', 'age', 'transaction_frequency',
            'amount_zscore', 'is_night_transaction', 'is_weekend_transaction',
            'is_amount_outlier', 'risk_score', 'is_high_risk_merchant'
        ]
        
        # Handle categorical variables
        categorical_cols = ['category']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if is_training:
                    le = LabelEncoder()
                    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        mask = df_processed[col].isin(le.classes_)
                        df_processed.loc[~mask, col] = le.classes_[0]  # Use first class for unknown
                        df_processed[col + '_encoded'] = le.transform(df_processed[col])
                
                feature_cols.append(col + '_encoded')
        
        # Select only available features
        available_features = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_features]
        
        # Fill any missing values
        X = X.fillna(0)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def train(self):
        """Train the fraud detection model"""
        print("Loading and preparing data...")
        df = self.load_and_prepare_data()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
        
        # Prepare features and target
        X = self.preprocess_features(df, is_training=True)
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\\nModel Performance:")
        print(f"AUC Score: {auc_score:.3f}")
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if len(X.columns) > 0:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\\nTop 10 Feature Importance:")
            print(feature_importance.head(10))
        
        # Save model and preprocessors
        os.makedirs('data/models', exist_ok=True)
        joblib.dump(self.model, 'data/models/fraud_model.pkl')
        joblib.dump(self.scaler, 'data/models/scaler.pkl')
        joblib.dump(self.label_encoders, 'data/models/label_encoders.pkl')
        
        print("\\nModel saved to data/models/")
        
        return self.model, auc_score

if __name__ == "__main__":
    model_trainer = FraudDetectionModel()
    model, auc_score = model_trainer.train()
import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import yaml
import mlflow
import mlflow.sklearn

class FraudDetectionModel:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.features = self.config['model']['features']
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        conn = duckdb.connect(self.config['database']['path'])
        
        query = """
        SELECT 
            amount,
            merchant_category,
            hour_of_day,
            day_of_week,
            customer_age,
            customer_total_transactions as transaction_frequency,
            amount_zscore as amount_percentile,
            is_night_transaction,
            is_weekend_transaction,
            is_amount_outlier,
            merchant_risk_score,
            is_fraud
        FROM mart_fraud_detection
        """
        
        df = conn.execute(query).df()
        conn.close()
        
        return df
    
    def preprocess_features(self, df, is_training=True):
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['merchant_category']
        
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    mask = df_processed[col].isin(le.classes_)
                    df_processed.loc[~mask, col] = 'unknown'
                    
                    # Add 'unknown' to classes if not present
                    if 'unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    
                    df_processed[col] = le.transform(df_processed[col])
        
        # Select features
        feature_cols = [col for col in self.features if col in df_processed.columns]
        X = df_processed[feature_cols]
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if is_training:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X
    
    def train(self):
        print("Loading data...")
        df = self.load_data()
        
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
        
        # Start MLflow run
        with mlflow.start_run():
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
            
            # Log metrics to MLflow
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 10,
                "test_size": self.test_size
            })
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\\nFeature Importance:")
            print(feature_importance.head(10))
            
            # Save model and preprocessors
            os.makedirs('data/models', exist_ok=True)
            joblib.dump(self.model, 'data/models/fraud_model.pkl')
            joblib.dump(self.scaler, 'data/models/scaler.pkl')
            joblib.dump(self.label_encoders, 'data/models/label_encoders.pkl')
            
            # Log model to MLflow
            mlflow.sklearn.log_model(self.model, "fraud_detection_model")
            
            print("\\nModel saved to data/models/")
        
        return self.model, auc_score
    
    def predict(self, X):
        if self.model is None:
            self.load_model()
        
        X_processed = self.preprocess_features(X, is_training=False)
        predictions = self.model.predict_proba(X_processed)[:, 1]
        return predictions
    
    def load_model(self):
        self.model = joblib.load('data/models/fraud_model.pkl')
        self.scaler = joblib.load('data/models/scaler.pkl')
        self.label_encoders = joblib.load('data/models/label_encoders.pkl')

if __name__ == "__main__":
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./data/mlruns")
    
    model_trainer = FraudDetectionModel()
    model, auc_score = model_trainer.train()
# Real-Time Payment Fraud Detection Platform

## Overview
End-to-end analytics platform for detecting fraudulent payment transactions in real-time using machine learning and behavioral analysis.

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Data Sources  │ -> │   Ingestion  │ -> │   Raw Storage   │
│                 │    │   (Python)   │    │   (DuckDB)      │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Dashboard     │ <- │  Analytics   │ <- │ Transformation  │
│  (Streamlit)    │    │   Models     │    │     (dbt)       │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                    ┌─────────────────┐
                    │  ML Pipeline    │
                    │ (scikit-learn)  │
                    └─────────────────┘
```

## Features
- Real-time fraud detection with <100ms response time
- Customer behavior profiling and risk scoring
- Transaction anomaly detection
- Merchant risk assessment
- Interactive fraud monitoring dashboard
- Data quality validation and monitoring

## Tech Stack
- **Data Processing**: dbt, Python, DuckDB
- **Machine Learning**: scikit-learn, MLflow
- **Visualization**: Streamlit, Plotly
- **Data Quality**: Great Expectations
- **Testing**: pytest

## Quick Start
```bash
# Run the complete pipeline
python3 run_simple_pipeline.py

# Launch the dashboard
source venv/bin/activate && cd dashboard && streamlit run app_simple.py
```

## Manual Setup (Alternative)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_simple.txt

# Generate sample data
python src/ingestion/data_generator.py

# Train fraud detection model
python src/ml/train_model_simple.py

# Launch dashboard
cd dashboard && streamlit run app_simple.py
```

## Project Structure
```
fraud-detection-platform/
├── data/                    # Data storage
├── src/                     # Source code
├── dbt_project/            # dbt transformations
├── dashboard/              # Streamlit app
├── config/                 # Configuration files
├── tests/                  # Test suite
└── docs/                   # Documentation
```
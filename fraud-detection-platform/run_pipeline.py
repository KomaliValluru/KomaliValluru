#!/usr/bin/env python3
"""
End-to-end pipeline runner for the Fraud Detection Platform
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, cwd=None):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete fraud detection pipeline"""
    
    print("🔒 Starting Fraud Detection Platform Pipeline")
    print("This will run the complete end-to-end pipeline.")
    
    # Check if we're in the right directory
    if not Path("config/config.yaml").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Step 2: Generate sample data
    if not run_command("python src/ingestion/data_generator.py", "Generating sample transaction data"):
        print("Failed to generate sample data")
        sys.exit(1)
    
    # Step 3: Install dbt dependencies
    if not run_command("dbt deps", "Installing dbt dependencies", cwd="dbt_project"):
        print("Failed to install dbt dependencies")
        sys.exit(1)
    
    # Step 4: Run dbt transformations
    if not run_command("dbt run", "Running dbt transformations", cwd="dbt_project"):
        print("Failed to run dbt transformations")
        sys.exit(1)
    
    # Step 5: Run dbt tests
    if not run_command("dbt test", "Running dbt data quality tests", cwd="dbt_project"):
        print("Warning: Some dbt tests failed. Check data quality.")
    
    # Step 6: Train ML model
    if not run_command("python src/ml/train_model.py", "Training fraud detection model"):
        print("Failed to train ML model")
        sys.exit(1)
    
    # Step 7: Run data quality tests
    if not run_command("python -m pytest tests/ -v", "Running data quality tests"):
        print("Warning: Some data quality tests failed")
    
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Launch the dashboard: streamlit run dashboard/app.py")
    print("2. View dbt documentation: dbt docs generate && dbt docs serve (from dbt_project/)")
    print("3. Check MLflow UI: mlflow ui --backend-store-uri file:./data/mlruns")
    print("\nProject structure:")
    print("- Raw data: data/raw/")
    print("- Processed data: data/fraud_detection.duckdb")
    print("- Trained model: data/models/")
    print("- Dashboard: dashboard/app.py")

if __name__ == "__main__":
    main()
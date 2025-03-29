
#!/usr/bin/env python3
"""
Main script for the Gene Expression Data Analysis pipeline.
This script orchestrates the entire workflow from data preprocessing to model evaluation.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from feature_selection import select_features
from model_training import train_evaluate_model
from visualization import plot_results

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def main():
    print("Starting Gene Expression Data Analysis Pipeline...")
    
    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    # Check if the preprocessed data already exists
    if os.path.exists("data/preprocessed_data.xlsx"):
        print("Loading preprocessed data...")
        preprocessed_data = pd.read_excel("data/preprocessed_data.xlsx", index_col=0)
    else:
        print("Processing raw data...")
        # Assuming the data file is in the data directory
        data_path = "GSE156564_series_matrix.txt"
        preprocessed_data = preprocess_data(data_path)
        # Save preprocessed data
        preprocessed_data.to_excel("data/preprocessed_data.xlsx")
    
    # Step 2: Feature Selection
    print("\nStep 2: Selecting features...")
    features, target, selected_features, mutual_info_dict = select_features(preprocessed_data)
    
    # Step 3: Model Training and Evaluation
    print("\nStep 3: Training and evaluating model...")
    model, X_test, y_test, y_pred, mse, rmse = train_evaluate_model(features, target, selected_features)
    
    # Step 4: Visualize Results
    print("\nStep 4: Visualizing results...")
    plot_results(features, target, mutual_info_dict, X_test, y_test, y_pred)
    
    # Step 5: Save Results
    print("\nStep 5: Saving results...")
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': y_test - y_pred
    })
    results_df.to_excel("results/prediction_results.xlsx")
    
    # Print performance metrics
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"Root Mean Squared Error: {rmse:.5f}")
    
    print("\nAnalysis pipeline completed successfully!")

if __name__ == "__main__":
    main()

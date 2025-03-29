
#!/usr/bin/env python3
"""
Module for data visualization functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def plot_results(features, target, mutual_info_dict, X_test, y_test, y_pred):
    """
    Creates visualizations for the analysis results.
    
    Parameters:
        features (pd.DataFrame): Feature dataframe.
        target (pd.Series): Target variable.
        mutual_info_dict (dict): Dictionary of mutual information values.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target values.
        y_pred (np.ndarray): Predicted values.
    """
    # 1. Plot mutual information
    plt.figure(figsize=(12, 6))
    sorted_mutual_info = sorted(mutual_info_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    features_names = [item[0] for item in sorted_mutual_info]
    mi_values = [item[1] for item in sorted_mutual_info]
    
    sns.barplot(x=mi_values, y=features_names)
    plt.title('Top 10 Features by Mutual Information')
    plt.xlabel('Mutual Information')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('figures/mutual_information_top10.png')
    
    # 2. Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig('figures/actual_vs_predicted.png')
    
    # 3. Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('figures/residuals.png')
    
    # 4. Distribution of features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features.columns[:min(10, len(features.columns))]):
        plt.subplot(2, 5, i+1)
        sns.histplot(features[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('figures/feature_distributions.png')
    
    # 5. Target distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(target, kde=True)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figures/target_distribution.png')

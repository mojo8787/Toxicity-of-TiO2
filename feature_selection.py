
#!/usr/bin/env python3
"""
Module for feature selection functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.svm import SVR

def select_features(preprocessed_data, target_column='score', 
                    exclude_columns=['ProbeID', 'GeneSymbol'], 
                    n_features_to_select=5):
    """
    Selects important features from the preprocessed data.
    
    Parameters:
        preprocessed_data (pd.DataFrame): The preprocessed data.
        target_column (str): The column name of the target variable.
        exclude_columns (list): List of column names to exclude from features.
        n_features_to_select (int): Number of features to select with RFE.
        
    Returns:
        (pd.DataFrame, pd.Series, pd.Index, dict): Features, target, selected features, and mutual info.
    """
    # Ensure target column is in the data
    if target_column not in preprocessed_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    
    # Separate features and target
    exclude_columns.append(target_column)
    features = preprocessed_data.drop(exclude_columns, axis=1, errors='ignore')
    target = preprocessed_data[target_column]
    
    # 1. Mutual Information
    mutual_info = mutual_info_regression(features, target)
    mutual_info_dict = dict(zip(features.columns, mutual_info))
    print("Mutual Information:")
    for feature, info in sorted(mutual_info_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - '{feature}': {info:.3f}")
    
    # 2. Correlation Matrix
    correlation_matrix = features.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png')
    
    # 3. Recursive Feature Elimination (RFE)
    selector = RFE(SVR(kernel="linear"), n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(features, target)
    selected_features = features.columns[selector.support_]
    print("\nFeatures selected by RFE:")
    for feature in selected_features:
        print(f"  - {feature}")
        
    # Save feature importance to file
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Mutual_Information': mutual_info,
        'Selected_by_RFE': selector.support_
    })
    feature_importance_df.sort_values('Mutual_Information', ascending=False, inplace=True)
    feature_importance_df.to_excel('results/feature_importance.xlsx', index=False)
    
    return features, target, selected_features, mutual_info_dict

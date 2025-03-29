
#!/usr/bin/env python3
"""
Module for data preprocessing functions.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(data_path):
    """
    Preprocesses gene expression data by handling missing values, scaling, and normalizing.
    
    Parameters:
        data_path (str): The path to the data file containing the gene expression data.
        
    Returns:
        pd.DataFrame: Preprocessed gene expression data.
    """
    # Load the data
    # Skip initial metadata rows and transpose the DataFrame to have samples as rows
    data = pd.read_csv(data_path, sep="\t", comment="!", index_col=0).T
    
    # Identify and remove any 'null' values that are not numeric (e.g., 'NA', 'null', etc.)
    data.replace(["null", "NA", "N/A"], np.nan, inplace=True)
    
    # Convert all columns to numeric types
    data = data.apply(pd.to_numeric, errors="coerce")
    
    # Impute missing values using the mean of each column
    data.fillna(data.mean(), inplace=True)
    
    # Calculate summary statistics after imputation
    summary_stats = data.describe()
    summary_stats.to_excel("results/summary_statistics.xlsx")
    
    # Separate numerical and non-numerical columns if any
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) < len(data.columns):
        non_numerical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        data_numerical = data[numerical_cols]
        data_non_numerical = data[non_numerical_cols]
    else:
        data_numerical = data
        data_non_numerical = None
    
    # Scale the numerical data
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data_numerical), 
        columns=numerical_cols, 
        index=data.index
    )
    
    # Normalize the scaled data
    normalizer = MinMaxScaler()
    data_normalized = pd.DataFrame(
        normalizer.fit_transform(data_scaled), 
        columns=numerical_cols, 
        index=data.index
    )
    
    # Combine back with non-numerical columns if any exist
    if data_non_numerical is not None:
        final_data = pd.concat([data_normalized, data_non_numerical], axis=1)
    else:
        final_data = data_normalized
    
    return final_data

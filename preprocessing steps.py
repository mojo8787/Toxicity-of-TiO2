import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(excel_path, sheet_name):
    """
    Preprocesses gene expression data by handling missing values, scaling, and normalizing.
    
    Parameters:
        excel_path (str): The path to the Excel file containing the gene expression data.
        sheet_name (str): The name of the sheet containing the data in the Excel file.
        
    Returns:
        pd.DataFrame: Preprocessed gene expression data.
    """
    # Step 1: Load Data
    data = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
    
    # Separate numerical and non-numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numerical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    data_numerical = data[numerical_cols]
    data_non_numerical = data[non_numerical_cols]
    
    # Step 2: Handle Missing Values for Numerical Columns
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numerical), columns=numerical_cols, index=data.index)
    
    # Step 3: Scale Data
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=numerical_cols, index=data.index)
    
    # Step 4: Normalize Data
    normalizer = MinMaxScaler()
    data_normalized = pd.DataFrame(normalizer.fit_transform(data_scaled), columns=numerical_cols, index=data.index)
    
    # Combine back the non-numerical columns
    preprocessed_data = pd.concat([data_normalized, data_non_numerical], axis=1)
    
    return preprocessed_data

# Define the path and sheet name based on your specific case
data_path = "C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\GSE156564_Processed_data_NRCWE001.xlsx"
sheet_name = "1.control_0-NRCWE001"

# Perform preprocessing
preprocessed_data = preprocess_data(data_path, sheet_name)

# Save the preprocessed data to an Excel file
preprocessed_data.to_excel("C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\Preprocessed_Data.xlsx")


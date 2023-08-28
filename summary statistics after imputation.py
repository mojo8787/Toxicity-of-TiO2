import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
# Skip initial metadata rows and transpose the DataFrame to have samples as rows
data_path = "C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\GSE156564_series_matrix.txt"

data = pd.read_csv(data_path, sep="\t", comment="!", index_col=0).T

# Identify and remove any 'null' values that are not numeric (e.g., 'NA', 'null', etc.)
data.replace(["null", "NA", "N/A"], np.nan, inplace=True)

# Convert all columns to numeric types
data = data.apply(pd.to_numeric, errors="coerce")

# Impute missing values using the mean of each column
data.fillna(data.mean(), inplace=True)

# Summary statistics
print("Summary statistics after imputation:")
print(data.describe())

# Data visualization
# Plotting the distribution of values for the first 10 genes
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.ravel()
for i, col in enumerate(data.columns[:10]):
    axs[i].hist(data[col], bins=20, color='blue', alpha=0.7, label='Frequency')
    axs[i].set_title(f"Distribution of {col}")
    axs[i].set_xlabel("Expression Level")
    axs[i].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

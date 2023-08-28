
# Gene Expression Data Analysis

## Overview

This project aims to analyze gene expression data to identify features that are strong predictors for a specific target variable.
The steps involved are feature selection, model training, and evaluation.

## Python Dependencies

- scikit-learn
- numpy
- seaborn
- matplotlib
- pandas

## Script Overview

The Python script performs the following steps:

1. **Feature Selection**: 
    - Mutual Information is calculated for each feature against the target variable.
    - A correlation matrix is plotted for the features.
    - Recursive Feature Elimination (RFE) is applied to select important features.

2. **Train-Test Split**: 
    - The data is split into training and testing sets.

3. **Model Training and Evaluation**: 
    - A Support Vector Machine for Regression (SVR) is trained on the data.
    - The model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Results

1. **Mutual Information**: 
    - 'AveExpr': 0.095
    - 't-statistic': 2.184
    - 'P.Value': 1.031
    - 'adj.P.Val': 1.185
    - 'B-statistic': 1.506

2. **Features Selected by RFE**: 
    - All features were deemed important by RFE.

3. **MSE and RMSE**: 
    - Mean Squared Error: 0.00205
    - Root Mean Squared Error: 0.0453

## Next Steps

1. **Hyperparameter Tuning**: Tuning the hyperparameters of the SVR model.
2. **Cross-Validation**: Implementing k-fold cross-validation.
3. **Additional Models**: Trying out other regression models for comparison.
4. **Feature Engineering**: Exploring feature interactions or transformations.
5. **Domain-Specific Analysis**: Consultation with domain experts for biological context.


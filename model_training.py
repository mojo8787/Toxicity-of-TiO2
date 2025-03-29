
#!/usr/bin/env python3
"""
Module for model training and evaluation functions.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def train_evaluate_model(features, target, selected_features=None, test_size=0.3, random_state=42):
    """
    Trains and evaluates a Support Vector Regression model.
    
    Parameters:
        features (pd.DataFrame): Feature dataframe.
        target (pd.Series): Target variable.
        selected_features (pd.Index, optional): Selected features to use. If None, use all features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        (SVR, pd.DataFrame, pd.Series, np.ndarray, float, float): 
            Trained model, test features, test target, predictions, MSE, and RMSE.
    """
    # Use selected features if provided
    if selected_features is not None and len(selected_features) > 0:
        X = features[selected_features]
        print(f"Using {len(selected_features)} selected features for model training.")
    else:
        X = features
        print(f"Using all {X.shape[1]} features for model training.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state
    )
    
    # Define hyperparameter grid for SVR
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['linear', 'rbf', 'poly']
    }
    
    # Perform grid search with cross-validation
    print("Performing hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        SVR(), 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"  - Mean Squared Error: {mse:.5f}")
    print(f"  - Root Mean Squared Error: {rmse:.5f}")
    print(f"  - R² Score: {r2:.5f}")
    
    # Save cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_excel('results/cross_validation_results.xlsx', index=False)
    
    return best_model, X_test, y_test, y_pred, mse, rmse

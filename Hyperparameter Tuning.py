from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the preprocessed data
preprocessed_data = pd.read_excel("C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\Preprocessed_Data.xlsx", index_col=0)

# Separate features and target variable
features = preprocessed_data.drop(['score', 'ProbeID', 'GeneSymbol'], axis=1)
target = preprocessed_data['score']

# 1. Feature Selection
## 1.1 Mutual Information
mutual_info = mutual_info_regression(features, target)
mutual_info_dict = dict(zip(features.columns, mutual_info))
print("Mutual Information:", mutual_info_dict)

## 1.2 Correlation Matrix
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

## 1.3 Recursive Feature Elimination (RFE)
selector = RFE(SVR(kernel="linear"), n_features_to_select=5, step=1)
selector = selector.fit(features, target)
print("Features selected by RFE:", features.columns[selector.support_])

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 3. Model Training with Support Vector Machine for Regression
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Model Evaluation
y_pred = svm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# 4. Hyperparameter Tuning
param_grid = [
    {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1]},
    {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.01, 0.1, 1]},
]

grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best parameters found: ", best_params)
print("Best estimator found: ", best_estimator)

best_score = best_estimator.score(X_test, y_test)
print("Test set score with best parameters: ", best_score)

# Save the actual and predicted values to a DataFrame
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Save to an Excel file
result_df.to_excel("C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\Predictions.xlsx")

print("Predictions saved to 'Predictions.xlsx'")

from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_regression  # Note the change here
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the preprocessed data
preprocessed_data = pd.read_excel("C:\\Users\\Motasem Younis\\Desktop\\Criminal_case_against_K.D\\Preprocessed_Data.xlsx", index_col=0)

# Separate features and target variable
features = preprocessed_data.drop(['score', 'ProbeID', 'GeneSymbol'], axis=1)  # Replace with your actual non-numeric and target columns
target = preprocessed_data['score']  # Replace with your actual target column

# 1. Feature Selection
## 1.1 Mutual Information
mutual_info = mutual_info_regression(features, target)  # Note the change here
mutual_info_dict = dict(zip(features.columns, mutual_info))
print("Mutual Information:", mutual_info_dict)

## 1.2 Correlation Matrix
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

## 1.3 Recursive Feature Elimination (RFE)
svc = SVC(kernel="linear", C=1)
selector = RFE(svc, n_features_to_select=5, step=1)
selector = selector.fit(features, target)
print("Features selected by RFE:", features.columns[selector.support_])

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 3. Model Training with Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
score = svm_model.score(X_test, y_test)
print("SVM Test Score:", score)

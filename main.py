# ===============================
# 1. Import Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv('loan_approval_dataset.csv')

# ===============================
# 3. Basic Data Cleaning
# ===============================
df.drop(columns='loan_id', inplace=True)

print(df.shape)
print(df.isnull().sum())
print(df.dtypes)

# ===============================
# 4. Feature Engineering (Assets)
# ===============================
df['movable_assets'] = df['bank_assets_value'] + df['luxury_asset_value']
df['immovable_assets'] = df['residential_asset_value'] + df['commercial_asset_value']

df.drop(
    columns=[
        'bank_assets_value',
        'luxury_asset_value',
        'residential_asset_value',
        'commercial_asset_value'
    ],
    inplace=True
)

print(df.describe())

# ===============================
# 5. Exploratory Data Analysis
# ===============================

# Number of Dependents
sns.countplot(x='no_of_dependents', data=df)
plt.title('Number of Dependents')
plt.show()

# Income vs Education
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(x='education', y='income_annum', data=df, ax=ax[0])
sns.violinplot(x='education', y='income_annum', data=df, ax=ax[1])
plt.show()

# Self Employed vs Education
sns.countplot(x='self_employed', data=df, hue='education')
plt.title('Self Employed vs Education')
plt.show()

# Loan Amount vs Loan Tenure
sns.boxplot(x='loan_tenure', y='loan_amount', data=df)
plt.title('Loan Amount vs Loan Tenure')
plt.show()
import seaborn as sns
print(sns.__version__)


# CIBIL Score Distribution
sns.histplot(df['cibil_score'], bins=30, kde=True)
plt.title('CIBIL Score Distribution')
plt.show()

# Assets vs Loan Status
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(x='movable_assets', data=df, hue='loan_status', ax=ax[0])
ax[0].set_title('Movable Assets vs Loan Status')

sns.histplot(x='immovable_assets', data=df, hue='loan_status', ax=ax[1])
ax[1].set_title('Immovable Assets vs Loan Status')

plt.tight_layout()
plt.show()

# Assets vs Loan Amount
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x='movable_assets', y='loan_amount', data=df, ax=ax[0])
ax[0].set_title('Movable Assets vs Loan Amount')

sns.scatterplot(x='immovable_assets', y='loan_amount', data=df, ax=ax[1])
ax[1].set_title('Immovable Assets vs Loan Amount')

plt.tight_layout()
plt.show()

# ===============================
# 6. Encoding Categorical Variables
# ===============================
df['education'] = df['education'].map({'Not Graduate': 0, 'Graduate': 1})
df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})
df['loan_status'] = df['loan_status'].map({'Rejected': 0, 'Approved': 1})

# ===============================
# 7. Correlation Heatmap
# ===============================
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# ===============================
# 8. Train Test Split
# ===============================
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# 9. Model Building
# ===============================

# Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

# Random Forest
rfc = RandomForestClassifier(random_state=42, n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

# ===============================
# 10. Model Evaluation
# ===============================
print("Decision Tree Accuracy:", dtree.score(X_test, y_test))
print("Random Forest Accuracy:", rfc.score(X_test, y_test))

print("\nDecision Tree Classification Report\n")
print(classification_report(y_test, dtree_pred))

print("\nRandom Forest Classification Report\n")
print(classification_report(y_test, rfc_pred))

# Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(confusion_matrix(y_test, dtree_pred), annot=True, fmt='d', ax=ax[0])
ax[0].set_title('Decision Tree Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, fmt='d', ax=ax[1])
ax[1].set_title('Random Forest Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

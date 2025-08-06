import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load datasets
df_train = pd.read_csv("churn-bigml-80.csv")
df_test = pd.read_csv("churn-bigml-20.csv")

# Encode binary features
df_train['International plan'] = df_train['International plan'].map({'Yes': 1, 'No': 0})
df_train['Voice mail plan'] = df_train['Voice mail plan'].map({'Yes': 1, 'No': 0})
df_train['Churn'] = df_train['Churn'].astype(int)

# Drop less informative columns
df_train.drop(['State', 'Area code'], axis=1, inplace=True)

# Prepare features and target
X = df_train.drop('Churn', axis=1)
y = df_train['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import matplotlib.pyplot as plt
import seaborn as sns

df_eda = df_train.copy()
df_eda['Churn'] = y

# Churn distribution
sns.countplot(x='Churn', data=df_eda)
plt.title('Churn Distribution')
plt.show()

# Customer service calls vs. churn
sns.boxplot(x='Churn', y='Customer service calls', data=df_eda)
plt.title('Customer Service Calls by Churn')
plt.show()

# International plan vs. churn
sns.barplot(x='International plan', y='Churn', data=df_eda)
plt.title('Churn Rate by International Plan')
plt.show()

# Correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df_eda.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print("Logistic Regression Report:")
print(classification_report(y_val, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_val, lr.predict_proba(X_val)[:,1]))

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_val)
print("Decision Tree Report:")
print(classification_report(y_val, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_val, tree.predict_proba(X_val)[:,1]))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print("Logistic Regression Report:")
print(classification_report(y_val, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_val, lr.predict_proba(X_val)[:,1]))

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_val)
print("Decision Tree Report:")
print(classification_report(y_val, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_val, tree.predict_proba(X_val)[:,1]))

from lifelines import CoxPHFitter

# Add a 'tenure' column as time duration (simulate)
df_surv = df_train.copy()
df_surv['duration'] = df_surv['Account length']
df_surv['event'] = df_surv['Churn']

# Drop unused and scale again
df_surv.drop(['Churn'], axis=1, inplace=True)

# Fit Cox Model
cox = CoxPHFitter()
cox.fit(df_surv, duration_col='duration', event_col='event')
cox.print_summary()
cox.plot()

import joblib

joblib.dump(lr, "logistic_model.pkl")
joblib.dump(tree, "tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")


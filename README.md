# Churnex
End-to-end Customer Churn Prediction using Decision Trees, Logistic Regression, and Survival Analysis. Includes EDA, model evaluation, and an interactive Streamlit dashboard for churn insights and business decision support

# Churnex

Customer Churn Prediction using Statistical Learning

---

## 📌 Overview

This project implements an end-to-end pipeline for customer churn prediction using:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Survival Analysis (Cox Proportional Hazards)**

It includes data preprocessing, exploratory analysis, model evaluation, and an interactive **Streamlit dashboard** for business insights.

---

## 💡 Motivation & Goals

Churn is one of the most critical metrics for subscription-based businesses. This project was built to:
- Understand **why customers leave**, and how to **predict it in advance**
- Apply **statistical and machine learning techniques** to a real-world business scenario
- Demonstrate the ability to go from **raw data → insights → predictive models → actionable dashboards**
- Bridge **academic theory (survival analysis, classification models)** with **business application**
- Strengthen my skills in data science for admission into a master's program in this field

The goal is not just prediction — but **interpretability**, **business relevance**, and **deployability**.

---

## 🛠️ Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| **Imbalanced target variable (churn is rare)** | Used evaluation metrics beyond accuracy (ROC-AUC, F1), and stratified split |
| **Categorical variables (e.g., Yes/No plans)** | Applied binary encoding for plan types |
| **Survival modeling required time-to-event format** | Created a synthetic duration feature using `Account length` |
| **Feature scaling** | Applied `StandardScaler` to prepare for logistic regression |
| **Multiple models with different assumptions** | Implemented all models separately and compared their results analytically |
| **Building an interactive business tool** | Used **Streamlit** for lightweight deployment and dashboard creation |

---

## 📂 Dataset

- Source: [Kaggle Telecom Customer Churn Dataset](https://www.kaggle.com/datasets)
- Split: `churn-bigml-80.csv` (train), `churn-bigml-20.csv` (test)

---

## 🖼️ Visual Insights

### 📊 Churn Distribution
- ![Image Alt](https://github.com/sakshikhonde67/Churnex/blob/75ae787000bccb1ed6ff2711d0538ad61bd1eb60/Churn%20Distribution.png)

### ☎️ Customer Service Calls by Churn
- ![Image Alt](https://github.com/sakshikhonde67/Churnex/blob/75ae787000bccb1ed6ff2711d0538ad61bd1eb60/Churn%20Customer%20Service%20Calls.png).

### 🌐 Churn Rate by International Plan
- ![Image Alt](https://github.com/sakshikhonde67/Churnex/blob/75ae787000bccb1ed6ff2711d0538ad61bd1eb60/Churn%20Rate%20by%20International%20Plan.png).

### 🔥 Correlation Matrix
- ![Image Alt](https://github.com/sakshikhonde67/Churnex/blob/75ae787000bccb1ed6ff2711d0538ad61bd1eb60/Churn%20Correlation%20Matrix.png).

### ⏳ Survival Analysis - log(HR)
- ![Image Alt](https://github.com/sakshikhonde67/Churnex/blob/75ae787000bccb1ed6ff2711d0538ad61bd1eb60/Churn%20Log%20(hr).png).


## 📈 Models & Evaluation

| Model              | Metrics                                 |
|-------------------|-----------------------------------------|
| Logistic Regression | Accuracy, Precision, Recall, ROC-AUC  |
| Decision Tree       | Accuracy, F1 Score, Feature Importance |
| Cox Model           | Survival curves, Retention probabilities |

---

## 📊 Dashboard

The dashboard is built with **Streamlit**, featuring:
- Key churn KPIs
- Model-based churn probability
- Interactive filters
- Business recommendations

📦 Installation

```bash
pip install -r requirements.txt


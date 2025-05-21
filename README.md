# 🩺 Diabetes Prediction Using Machine Learning (BRFSS 2015)

This project uses multiple machine learning models to predict the likelihood of diabetes based on health indicators from the 2015 Behavioral Risk Factor Surveillance System (BRFSS). After evaluating and tuning several models, **LightGBM** was selected as the best performer in terms of **accuracy** and **F1-score**.

## 📁 Dataset

- **Source:** [Kaggle - Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_binary_health_indicators_BRFSS2015.csv)
- **File:** `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Target variable:** `Diabetes_binary` (0 = No diabetes, 1 = Diabetes or Pre-Diabetes)

## ⚙️ Objective

To build predictive models using BRFSS data to identify individuals at risk of diabetes, enabling early intervention and supporting public health strategies.

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values, data imbalance, and feature distributions.
- Found that the dataset is highly imbalanced: most individuals are non-diabetic.
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance.

---

## 🤖 Models Evaluated

The following models were trained and tuned using **GridSearchCV**:

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM ✅ (Best model)
- MLP Classifier

Each model was evaluated using:

- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion Matrix
- Classification Report

---

## 🧪 Model Tuning

Grid Search was applied on a **subsampled** dataset to reduce computation time. Final models were then retrained on the full resampled dataset (`X_train_resampled`, `y_train_resampled`).

The best performing model was:

```text
✅ LightGBM
Best Parameters: { 'num_leaves': 50, 'objective': 'binary' }
Test Accuracy: 0.864219
ROC AUC: 0.823866
```

> MLP Classifier performed best in recall, which may be important in public health screening applications.

## 🩺 Public Health Implications
- Recommendation: Focus on early detection through public health screenings targeting high-risk individuals (e.g., with high BMI, hypertension).
- Promote lifestyle interventions (exercise, weight loss) to reduce diabetes risk.
- Target education and preventive care to populations with the highest predicted risk.

## 🧾 Requirements
- pandas, numpy
- scikit-learn
- imbalanced-learn
- lightgbm, xgboost
- matplotlib, seaborn
- shap

This project demonstrates a simple machine learning pipeline to predict diabetes using the Pima Indians Diabetes Dataset. The workflow includes data loading, preprocessing, train-test splitting, model training using XGBoost, and generating predictions.

📌 Project Overview

The goal of this project is to build a binary classification model that predicts whether a patient has diabetes based on medical attributes.

The pipeline is modular and includes:

Data loading using NumPy
Train-test split using scikit-learn
Model training using XGBoost Classifier
Prediction on unseen test data
📂 Dataset

The dataset used is:
Pima Indians Diabetes Dataset

File: pima-indians-diabetes.csv
Format: CSV
Features: 8 medical attributes
Target: 1 (diabetic) / 0 (non-diabetic)

Typical features include:
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age

⚙️ Installation
Make sure you have the required libraries installed:
pip install numpy scikit-learn xgboost

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:19:23 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_hc = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Advanced ML Algorithms/Healthcare_Dataset_Preprocessednew.csv')
df_hc.columns
df_hc.dtypes
df_hc.shape
df_hc.head()

missing_values = df_hc.isnull().sum()
print(missing_values)

for col in ['Diet_Type__Vegan', 'Diet_Type__Vegetarian', 'Blood_Group_AB', 'Blood_Group_B','Blood_Group_O']: 
    if df_hc[col].dtype == 'bool':
        df_hc[col] = df_hc[col].astype(int)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

X = df_hc.drop(columns=['Target'])
y = df_hc['Target']

# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize models
random_forest = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1)
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)

# Train and evaluate Random Forest
n_estimators_range = range(10, 210, 10)
rf_train_scores = []
rf_oob_scores = []
rf_test_scores = []

for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    rf_train_scores.append(rf.score(X_train_scaled, y_train))
    rf_oob_scores.append(rf.oob_score_)
    rf_test_scores.append(rf.score(X_test_scaled, y_test))

# Train and evaluate Bagging with Decision Trees
bagging.fit(X_train_scaled, y_train)
bagging_train_score = bagging.score(X_train_scaled, y_train)
bagging_test_score = bagging.score(X_test_scaled, y_test)

# Train and evaluate Decision Tree
decision_tree.fit(X_train_scaled, y_train)
decision_tree_train_score = decision_tree.score(X_train_scaled, y_train)
decision_tree_test_score = decision_tree.score(X_test_scaled, y_test)


print("Performance Metrics:")
print(f"Random Forest - OOB Score (Best): {max(rf_oob_scores):.4f}")
print(f"Random Forest - Test Accuracy (Best): {max(rf_test_scores):.4f}")
print(f"Bagging - Train Accuracy: {bagging_train_score:.4f}, Test Accuracy: {bagging_test_score:.4f}")
print(f"Decision Tree - Train Accuracy: {decision_tree_train_score:.4f}, Test Accuracy: {decision_tree_test_score:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, rf_train_scores, label='Random Forest - Train Score', marker='o')
plt.plot(n_estimators_range, rf_oob_scores, label='Random Forest - OOB Score', marker='x')
plt.plot(n_estimators_range, rf_test_scores, label='Random Forest - Test Score', marker='s')
plt.axhline(y=bagging_test_score, color='green', linestyle='--', label='Bagging - Test Score')
plt.axhline(y=decision_tree_test_score, color='red', linestyle='--', label='Decision Tree - Test Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------------------------------------------------------
# Adding Hyperparameter tuning
# -----------------------------------------------------------------------------

# Random Forest Tuning

rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_scaled, y_train)

# Best parameters and score for Random Forest
rf_best = rf_grid.best_estimator_
rf_train_score = rf_best.score(X_train_scaled, y_train)
rf_test_score = rf_best.score(X_test_scaled, y_test)


# Bagging with Decision Tree Tuning

bagging_param_grid = {
    'n_estimators': [50, 100, 150],
    'base_estimator__max_depth': [10, 20, None],
    'base_estimator__min_samples_split': [2, 5, 10],
    'base_estimator__min_samples_leaf': [1, 2, 4],
}
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
bagging_grid = GridSearchCV(estimator=bagging, param_grid=bagging_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
bagging_grid.fit(X_train_scaled, y_train)

# Best parameters and score for Bagging
bagging_best = bagging_grid.best_estimator_
bagging_train_score = bagging_best.score(X_train_scaled, y_train)
bagging_test_score = bagging_best.score(X_test_scaled, y_test)


# Decision Tree Tuning

dt_param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(estimator=dt, param_grid=dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
dt_grid.fit(X_train_scaled, y_train)

# Best parameters and score for Decision Tree
dt_best = dt_grid.best_estimator_
dt_train_score = dt_best.score(X_train_scaled, y_train)
dt_test_score = dt_best.score(X_test_scaled, y_test)


print("\n--- Model Comparison ---")
print(f"Random Forest - Best Params: {rf_grid.best_params_}, Train Score: {rf_train_score:.4f}, Test Score: {rf_test_score:.4f}")
print(f"Bagging - Best Params: {bagging_grid.best_params_}, Train Score: {bagging_train_score:.4f}, Test Score: {bagging_test_score:.4f}")
print(f"Decision Tree - Best Params: {dt_grid.best_params_}, Train Score: {dt_train_score:.4f}, Test Score: {dt_test_score:.4f}")

# Visualization

models = ['Random Forest', 'Bagging', 'Decision Tree']
train_scores = [rf_train_score, bagging_train_score, dt_train_score]
test_scores = [rf_test_score, bagging_test_score, dt_test_score]

plt.figure(figsize=(10, 6))
x = range(len(models))
plt.bar(x, train_scores, width=0.4, label='Train Score', align='center')
plt.bar(x, test_scores, width=0.4, label='Test Score', align='edge')
plt.xticks(x, models)
plt.ylim(0.7, 1.0)
plt.ylabel('Accuracy Score')
plt.title('Model Comparison: Train vs Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()















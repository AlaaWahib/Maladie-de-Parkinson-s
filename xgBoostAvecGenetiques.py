# -*- coding: utf-8 -*-
"""
Parkinson's Detection using Genetic Algorithm for Feature Selection + XGBoost
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pygad

# === 1. SET RANDOM SEEDS FOR REPRODUCIBILITY ===
random.seed(42)
np.random.seed(42)

# === 2. LOAD AND PREPARE DATA ===
df = pd.read_csv(r"C:\Users\user\Downloads\Dataprojet\telemonitoring\parkinsons_updrs.data")
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

# Create binary label based on median of total_UPDRS
df['label'] = (df['total_UPDRS'] > df['total_UPDRS'].median()).astype(int)

# Retain feature names for later
feature_columns = df.drop(columns=['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'label']).columns
features = df[feature_columns].values

# Normalize features to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaled = scaler.fit_transform(features)
y = df['label'].values
n_features = x_scaled.shape[1]

# === 3. DEFINE FITNESS FUNCTION FOR GA ===
def fitness_func(ga_instance, solution, solution_idx):
    selected_indices = [i for i, val in enumerate(solution) if val == 1]
    if len(selected_indices) == 0:
        return 0
    x_selected = x_scaled[:, selected_indices]

    # Lightweight model for faster evaluation
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    scores = cross_val_score(model, x_selected, y, cv=3, scoring='accuracy')
    return scores.mean()

# === 4. CONFIGURE AND RUN GENETIC ALGORITHM ===
ga_instance = pygad.GA(
    num_generations=6,
    num_parents_mating=4,
    sol_per_pop=10,
    num_genes=n_features,
    gene_space=[0, 1],
    gene_type=int,
    mutation_percent_genes=15,
    crossover_type="single_point",
    mutation_type="random",
    fitness_func=fitness_func,
    random_seed=42
)

ga_instance.run()

# === 5. EXTRACT BEST FEATURE SELECTION ===
solution, solution_fitness, _ = ga_instance.best_solution()
selected_indices = [i for i, val in enumerate(solution) if val == 1]
selected_features = [feature_columns[i] for i in selected_indices]

print(f"\n Selected Features: {selected_features}")
print(f" GA Fitness Score (Accuracy): {solution_fitness:.4f}")

x_selected = x_scaled[:, selected_indices]

# === 6. FINAL TRAINING WITH XGBOOST ===
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=7)

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(x_train, y_train)

# === 7. EVALUATE FINAL MODEL ===
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n Final Accuracy with XGBoost: {accuracy:.2f}%")

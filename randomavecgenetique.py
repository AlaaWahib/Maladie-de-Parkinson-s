# -*- coding: utf-8 -*-
"""
Parkinson Detection with Genetic Algorithm (Reproducible Version) - No Tkinter GUI
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pygad

# === 1. SET RANDOM SEEDS ===
random.seed(42)
np.random.seed(42)

# === 2. LOAD DATA ===
# Assurez-vous que le chemin d'accès au fichier est correct pour votre environnement
df = pd.read_csv(r"C:\\Users\\user\\Downloads\\Dataprojet\\parkinsons.data")
features = df.drop(['name', 'status'], axis=1)
target = df['status']

# Normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(features)

# === 3. GENETIC ALGORITHM FOR FEATURE SELECTION ===
X = features_scaled
y = target.values
n_features = X.shape[1]

def fitness_func(ga_instance, solution, solution_idx):
    selected_indices = [i for i, val in enumerate(solution) if val == 1]
    if len(selected_indices) == 0:
        return 0
    X_selected = X[:, selected_indices]
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return np.mean(scores)

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=12,
    num_genes=n_features,
    gene_space=[0, 1],
    gene_type=int,
    mutation_percent_genes=10,
    crossover_type="single_point",
    mutation_type="random",
    random_seed=42  # Fix for reproducibility
)

print("--- Lancement de l'algorithme génétique pour la sélection de caractéristiques ---")
ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
selected_indices = [i for i, val in enumerate(solution) if val == 1]

print(f"\nMeilleure solution (sélection de caractéristiques) : {solution}")
print(f"Fitness de la meilleure solution : {solution_fitness:.4f}")
print(f"Indices des caractéristiques sélectionnées : {selected_indices}")

# === AJOUT POUR AFFICHER LES NOMS DES CARACTÉRISTIQUES SÉLECTIONNÉES ===
all_feature_names = features.columns.tolist()
selected_feature_names = [all_feature_names[i] for i in selected_indices]
print(f"Noms des caractéristiques sélectionnées par l'AG : {selected_feature_names}")
# =====================================================================

# === 4. TRAIN FINAL MODEL ===
features_ga_selected = features.iloc[:, selected_indices]
feature_names = features_ga_selected.columns.tolist() # Cette ligne est déjà là et sera la même que selected_feature_names

x_train, x_test, y_train, y_test = train_test_split(
    features_ga_selected, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
print("\n--- Entraînement du modèle RandomForest ---")
model.fit(x_train, y_train)
print("Modèle entraîné avec succès.")

# === 5. PLOT TREES ===
print("\n--- Génération du graphique des arbres de décision ---")
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3), dpi=100)
for index in range(5):
    tree.plot_tree(
        model.estimators_[index],
        feature_names=feature_names,
        class_names=['No Parkinson', 'Parkinson'],
        filled=True,
        ax=axes[index]
    )
    axes[index].set_title(f'Estimator {index + 1}', fontsize=10)
fig.tight_layout()
fig.savefig('Random_Forest_5_Trees.png')
plt.close()
print("Graphique 'Random_Forest_5_Trees.png' généré.")

# === 6. EVALUATE MODEL ===
y_pred = model.predict(x_test)
print("\n--- Évaluation du modèle ---")
print(f"Accuracy                       : {accuracy_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error            : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error        : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

print("\n--- Fin du script ---")
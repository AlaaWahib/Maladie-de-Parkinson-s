# -*- coding: utf-8 -*-
"""
XGBoost - Détection Parkinson + interface Tkinter
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import messagebox
from xgboost import XGBClassifier

# Chargement des données
df = pd.read_csv(r"C:\Users\user\Downloads\Dataprojet\parkinsons.data")

# Colonnes utilisées et non utilisées
excluded_columns = ['name', 'status']
features = df.drop(columns=excluded_columns)
target = df['status']
feature_names = features.columns.tolist()

print(" Colonnes utilisées dans le modèle (features) :")
print(feature_names)
print("\n Colonnes exclues (non utilisées) :")
print(excluded_columns)

# Normalisation
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(features)

# Split
x_train, x_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=10)

# Modèle XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2)
model.fit(x_train, y_train)

# Prédictions
y_pred = model.predict(x_test)

# Évaluation
print("\n--- Évaluation du modèle ---")
print(f"Accuracy               : {accuracy_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error    : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Interface Tkinter
def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        if len(values) != len(feature_names):
            raise ValueError("Nombre incorrect de valeurs.")

        input_array = np.array(values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        result_label.config(
            text="Résultat : Parkinson détecté" if prediction == 1 else "Résultat : Pas de Parkinson",
            fg="red" if prediction == 1 else "green"
        )
    except Exception as e:
        messagebox.showerror("Erreur", f"Entrée invalide : {e}")

def show_metrics():
    messagebox.showinfo(
        "Performance du modèle",
        f"Accuracy : {accuracy_score(y_test, y_pred):.4f}\nMAE     : {mean_absolute_error(y_test, y_pred):.2f}\nRMSE    : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}"
    )

def clear_entries():
    for entry in entries:
        entry.delete(0, tk.END)
    result_label.config(text="")

# GUI
root = tk.Tk()
root.title("Détection de la maladie de Parkinson - XGBoost")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

entries = []
for i, feature in enumerate(feature_names):
    lbl = tk.Label(frame, text=feature)
    lbl.grid(row=i, column=0, sticky="w")
    ent = tk.Entry(frame, width=10)
    ent.grid(row=i, column=1)
    entries.append(ent)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

btn_predict = tk.Button(button_frame, text="Prédire", command=predict, bg="#4CAF50", fg="white")
btn_predict.pack(side=tk.LEFT, padx=5)

btn_clear = tk.Button(button_frame, text="Effacer", command=clear_entries, bg="#f44336", fg="white")
btn_clear.pack(side=tk.LEFT, padx=5)

btn_metrics = tk.Button(root, text="Afficher les performances", command=show_metrics)
btn_metrics.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()

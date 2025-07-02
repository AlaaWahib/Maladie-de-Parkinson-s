# -*- coding: utf-8 -*-
"""
XGBoost Classifier - Parkinsons Dataset
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Lire le fichier CSV
df = pd.read_csv('C:/Users/user/Downloads/Dataprojet/telemonitoring/parkinsons_updrs.data')
df.columns = df.columns.str.strip()

# Créer étiquette binaire
df['label'] = (df['total_UPDRS'] > df['total_UPDRS'].median()).astype(int)

# Colonnes exclues
excluded_columns = ['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'label']
print("Colonnes exclues :")
print(excluded_columns)

# Features utilisées
feature_columns = df.drop(columns=excluded_columns).columns
print("\nFeatures utilisées :")
print(list(feature_columns))

# Données
features = df[feature_columns].values
labels = df['label'].values

# Normalisation
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Modèle XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=7)
model.fit(x_train, y_train)

# Prédictions
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

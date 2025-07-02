import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# Lire le fichier CSV
df = pd.read_csv('C:/Users/user/Downloads/Dataprojet/telemonitoring/parkinsons_updrs.data')
df.head()
# Nettoyage éventuel des colonnes (si noms avec espaces)
df.columns = df.columns.str.strip()

# Créer une étiquette binaire selon le score total_UPDRS (classification)
# Ici on sépare les patients au-dessus ou en dessous de la médiane
df['label'] = (df['total_UPDRS'] > df['total_UPDRS'].median()).astype(int)

# Sélection des caractéristiques (on retire les colonnes non utiles)
features = df.drop(columns=['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'label'])
feature_names = features.columns.tolist()
labels = df['label'].values

# Mise à l'échelle des données entre -1 et 1
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(features)
y = labels

# Séparation entraînement / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Fonction pour sélectionner les meilleures features
def select_best_features(x_train, y_train, feature_names, threshold='median'):
    # D'abord entraîner un modèle initial pour la sélection de features
    initial_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=7)
    initial_model.fit(x_train, y_train)
    
    # Créer un sélecteur basé sur l'importance des features
    selector = SelectFromModel(initial_model, threshold=threshold, prefit=True)
    
    # Obtenir les indices des features sélectionnées
    feature_indices = selector.get_support()
    
    # Filtrer les features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_indices[i]]
    
    # Transformer les ensembles de données
    x_train_selected = selector.transform(x_train)
    
    return selected_features, feature_indices, selector, initial_model

# Sélectionner les meilleures features
selected_features, feature_indices, selector, initial_model = select_best_features(x_train, y_train, feature_names)

print(f"Nombre de features original: {len(feature_names)}")
print(f"Nombre de features sélectionnées: {len(selected_features)}")
print("Features sélectionnées:", selected_features)

# Visualisation de l'importance des features
def plot_feature_importances(model, feature_names, selected_features, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Importance des caractéristiques")
    
    # Créer une liste de couleurs (vert pour les sélectionnées, rouge pour les autres)
    colors = ['green' if feature_names[i] in selected_features else 'red' for i in indices[:top_n]]
    
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], color=colors[::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel("Importance relative")
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()

plot_feature_importances(initial_model, feature_names, selected_features)

# Entraînement du modèle XGBoost avec les features sélectionnées
x_train_selected = selector.transform(x_train)
x_test_selected = selector.transform(x_test)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=7)
model.fit(x_train_selected, y_train)

# Prédictions et évaluation
y_pred = model.predict(x_test_selected)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy avec features sélectionnées: {accuracy:.2f}%")

# Rapport de classification détaillé
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=["Bas UPDRS", "Haut UPDRS"]))

# Pour comparaison, entraînement avec toutes les features
full_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=7)
full_model.fit(x_train, y_train)
full_y_pred = full_model.predict(x_test)
full_accuracy = accuracy_score(y_test, full_y_pred) * 100
print(f"\nAccuracy avec toutes les features: {full_accuracy:.2f}%")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# Lire le fichier CSV
df = pd.read_csv('C:/Users/user/Downloads/Dataprojet/telemonitoring/parkinsons_updrs.data')
df.columns = df.columns.str.strip()

# Créer une étiquette binaire selon le score total_UPDRS
df['label'] = (df['total_UPDRS'] > df['total_UPDRS'].median()).astype(int)

# Sélection des caractéristiques
features = df.drop(columns=['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'label'])
feature_names = features.columns.tolist()
labels = df['label'].values

# Mise à l'échelle
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Fonction de sélection des meilleures features
def select_best_features(x_train, y_train, feature_names, threshold='median'):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        random_state=7,
        class_weight='balanced'
    )
    rf_model.fit(x_train, y_train)

    selector = SelectFromModel(rf_model, threshold=threshold, prefit=True)
    feature_indices = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_indices[i]]
    x_train_selected = selector.transform(x_train)
    return selected_features, feature_indices, selector, rf_model

# Sélection des features
selected_features, feature_indices, selector, initial_model = select_best_features(x_train, y_train, feature_names)

print(f"Nombre de features original: {len(feature_names)}")
print(f"Nombre de features sélectionnées: {len(selected_features)}")
print("Features sélectionnées:", selected_features)

# Visualisation des importances
def plot_feature_importances(model, feature_names, selected_features, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Importance des caractéristiques")
    colors = ['green' if feature_names[i] in selected_features else 'red' for i in indices[:top_n]]
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], color=colors[::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel("Importance relative")
    plt.tight_layout()
    plt.savefig('feature_importances_rf.png')
    plt.show()

plot_feature_importances(initial_model, feature_names, selected_features)

# Transformation des données
x_train_selected = selector.transform(x_train)
x_test_selected = selector.transform(x_test)

# Modèle Random Forest avec 9 paramètres
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=False,
    random_state=7,
    class_weight='balanced'
)
model.fit(x_train_selected, y_train)

# Évaluation du modèle réduit
y_pred = model.predict(x_test_selected)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy avec features sélectionnées: {accuracy:.2f}%")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=["Bas UPDRS", "Haut UPDRS"]))

# Comparaison avec toutes les features (sans sélection)
full_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=False,
    random_state=7,
    class_weight='balanced'
)
full_model.fit(x_train, y_train)
full_y_pred = full_model.predict(x_test)
full_accuracy = accuracy_score(y_test, full_y_pred) * 100
print(f"\nAccuracy avec toutes les features: {full_accuracy:.2f}%")

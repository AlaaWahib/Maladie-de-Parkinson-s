# -*- coding: utf-8 -*-
"""
Parkinson Detection with VAE + Genetic Algorithm + XGBoost
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pygad
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import xgboost as xgb

# === 1. SET RANDOM SEEDS ===
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# === 2. LOAD DATA ===
file_path = os.path.join(os.path.expanduser("~"), "Desktop", "pfe", "parkinsons.data")
df = pd.read_csv(file_path)
features = df.drop(['name', 'status'], axis=1)
target = df['status']

# Normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(features)

# === 3. DEFINE VARIATIONAL AUTOENCODER ===
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=10):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='tanh')(x)
    decoder = Model(latent_inputs, outputs, name="decoder")

    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def compile(self, optimizer):
            super(VAE, self).compile()
            self.optimizer = optimizer
            self.loss_fn = tf.keras.losses.MeanSquaredError()

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                recon_loss = self.loss_fn(data, reconstruction)
                kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
                total_loss = recon_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": total_loss}

    vae = VAE(encoder, decoder)
    return vae, encoder

# === 4. TRAIN VAE AND TRANSFORM DATA ===
input_dim = features_scaled.shape[1]
latent_dim = 10
vae, encoder = build_vae(input_dim, latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(features_scaled, epochs=50, batch_size=16, verbose=0)

z_mean, _, _ = encoder.predict(features_scaled)
X = z_mean
y = target.values
n_features = X.shape[1]

# === 5. GENETIC ALGORITHM FOR FEATURE SELECTION ===
def fitness_func(ga_instance, solution, solution_idx):
    selected_indices = [i for i, val in enumerate(solution) if val == 1]
    if len(selected_indices) == 0:
        return 0
    X_selected = X[:, selected_indices]
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
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
    random_seed=42
)

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
selected_indices = [i for i, val in enumerate(solution) if val == 1]

# Print number of selected features
print(f"Number of selected latent features: {len(selected_indices)}")
print(f"Selected indices: {selected_indices}")

# === 6. TRAIN FINAL MODEL ===
X_selected = X[:, selected_indices]
x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("\n--- Évaluation du modèle ---")
print(f"Accuracy               : {accuracy_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error    : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

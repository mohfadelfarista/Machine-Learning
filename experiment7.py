# ============================================
# LEMBAR KERJA 7 — EKSPERIMEN & PELAPORAN
# ============================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Pengaturan awal
# --------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)

# --------------------------------------------
# 2. Load dataset
# --------------------------------------------
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')
y_test = pd.read_csv('y_test.csv')

# Ubah ke numpy array
X_train, X_val, X_test = X_train.values, X_val.values, X_test.values
y_train, y_val, y_test = y_train.values, y_val.values, y_test.values

# --------------------------------------------
# 3. Buat model (ubah jumlah neuron di sini)
# --------------------------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# --------------------------------------------
# 4. Pilih optimizer (bandingkan Adam vs SGD)
# --------------------------------------------
optimizer = Adam(learning_rate=0.001)
# optimizer = SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------------------------
# 5. Training model
# --------------------------------------------
es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es],
    verbose=1
)

# --------------------------------------------
# 6. Evaluasi model
# --------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# --------------------------------------------
# 7. Tambahan metrik F1 dan AUC
# --------------------------------------------
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# --------------------------------------------
# 8. Simpan grafik learning curve
# --------------------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve_experiment.png", dpi=120)
plt.close()

print("\n📊 Grafik 'learning_curve_experiment.png' telah disimpan.")
print("Eksperimen selesai ✅")

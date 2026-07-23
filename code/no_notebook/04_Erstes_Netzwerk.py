# %% [markdown]
# # 5. Stunde: Erstes Neuronales Netzwerk trainieren
# ## Mehrschichtiges Fully Connected Network
# 
# In diesem Notebook trainieren wir unser erstes echtes neuronales Netzwerk zur Handschrifterkennung.

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    print("Cannot import tensorflow: using jax")
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    from keras import layers
import time

print("✓ Alle Bibliotheken importiert")

# %% [markdown]
# ## 1. Daten laden und vorbereiten

# %%
# MNIST-Daten laden
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Rohdaten geladen:")
print(f"  Training: {x_train.shape}, Labels: {y_train.shape}")
print(f"  Test: {x_test.shape}, Labels: {y_test.shape}")

# Daten vorbereiten: Reshape und Normalisieren
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat = x_test.reshape(-1, 784) / 255.0

# One-Hot Encoding der Labels
y_train_encoded = keras.utils.to_categorical(y_train, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

print("\nDaten vorbereitet:")
print(f"  Training Bilder: {x_train_flat.shape}")
print(f"  Training Labels (One-Hot): {y_train_encoded.shape}")
print(f"  Pixel-Bereich: [{x_train_flat.min():.2f}, {x_train_flat.max():.2f}]")

# %% [markdown]
# ## 2. Modell-Architektur definieren

# %%
def build_model(
    input_size=784,
    hidden_layers=[128, 64],
    activation="relu",
    dropout_rate=0.0,
    output_classes=10,
    output_activation="softmax",
    learning_rate=0.001
):
    """
    Erstellt ein vollständig verbundenes neuronales Netzwerk.

    Parameters
    ----------
    input_size:
        Anzahl Eingabewerte (MNIST: 784 Pixel)

    hidden_layers:
        Liste mit Anzahl Neuronen pro versteckter Schicht

    activation:
        Aktivierungsfunktion der versteckten Schichten

    dropout_rate:
        Anteil der ausgeschalteten Neuronen während Training

    output_classes:
        Anzahl Klassen

    learning_rate:
        Schrittweite beim Lernen
    """


    model = keras.Sequential()


    # Input + Hidden Layers

    for i, neurons in enumerate(hidden_layers):

        if i == 0:
            model.add( layers.Dense( neurons, activation=activation, input_shape=(input_size,), name=f"hidden_layer_{i+1}"))
        else:
            model.add( layers.Dense( neurons, activation=activation, name=f"hidden_layer_{i+1}"))

        if dropout_rate > 0:
            model.add( layers.Dropout( dropout_rate, name=f"dropout_{i+1}"))

    # Output layer
    model.add( layers.Dense( output_classes, activation=output_activation, name="output_layer"))
    optimizer = keras.optimizers.Adam( learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        # loss="sparse_categorical_crossentropy",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model

# %% [markdown]
# ### Different Models

# %%
model_linear = build_model(
    hidden_layers=[],
    activation=None
)

model_linear.summary()

# %%
model_medium = build_model(
    hidden_layers=[
        128,
        64
    ],
    activation="relu",
    dropout_rate=0.1
)

model_medium.summary()

# %%
model_deep = build_model(
    hidden_layers=[
        512,
        256,
        128
    ],
    activation="relu",
    dropout_rate=0.2
)

model_deep.summary()

# %%
# Modell erstellen: 784 → 512 → 256 → 128 → 10 (verbesserte Architektur)
# model = keras.Sequential([
#     layers.Dense(512, activation='relu', input_shape=(784,), name='input_layer_0'),
#     layers.Dropout(0.2, name='dropout_1'),
#     layers.Dense(256, activation='relu', name='hidden_layer_1'),
#     layers.Dropout(0.2, name='dropout_2'),
#     layers.Dense(128, activation='relu', name='hidden_layer_2'),
#     layers.Dropout(0.2, name='dropout_3'),
#     layers.Dense(10, activation='softmax', name='output_layer')
# ])

# # Modell Zusammenfassung
# model.summary()
# 
# Parameter zählen
# total_params = model.count_params()
# print(f"\nGesamte Parameter: {total_params:,}")

# %% [markdown]
# ## 3. Modell kompilieren

# %%
# Optimierer mit Learning Rate
# optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Modell kompilieren
# model.compile(
    # optimizer=optimizer,
    # loss='categorical_crossentropy',
    # metrics=['accuracy']
# )

# print("✓ Modell kompiliert mit:")
# print("  - Optimizer: Adam (Learning Rate = 0.001)")
# print("  - Loss: Categorical Crossentropy")
# print("  - Metriken: Accuracy")

# %%
model = model_deep

# %% [markdown]
# ## 4. Modell trainieren

# %%
# Training mit Validierungsdaten
print("Training startet...\n")
start_time = time.time()

history = model.fit(
    x_train_flat, y_train_encoded,
    epochs=15,
    batch_size=32,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✓ Training beendet in {training_time:.2f} Sekunden")

# %% [markdown]
# ## 5. Trainings-Verlauf visualisieren

# %%
# Trainings-Verlauf plotten
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss-Plot
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
try: axes[0].plot(history.history['val_loss'], label='Validierung Loss', linewidth=2)
except: pass
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss über Zeit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy-Plot
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
try: axes[1].plot(history.history['val_accuracy'], label='Validierung Accuracy', linewidth=2)
except: pass
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Genauigkeit über Zeit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.pause(0.3)
plt.pause(0.3)
plt.show(block=False)

# Letzte Werte anzeigen
print(f"Finaler Training Loss: {history.history['loss'][-1]:.4f}")
try: print(f"Finaler Validierung Loss: {history.history['val_loss'][-1]:.4f}")
except: pass
print(f"Finaler Training Accuracy: {history.history['accuracy'][-1]:.4f}")
try: print(f"Finaler Validierung Accuracy: {history.history['val_accuracy'][-1]:.4f}")
except: pass

# %% [markdown]
# ## 6. Auf Testset evaluieren

# %%
# Testset evaluieren
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test_encoded, verbose=0)

print("\n" + "="*50)
print("TESTSET ERGEBNISSE")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("="*50)

# %% [markdown]
# ## 7. Einzelne Vorhersagen testen

# %%
# 10 zufällige Test-Bilder vorhersagen
indices = np.random.choice(len(x_test), 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, idx in enumerate(indices):
    # Vorhersage
    prediction = model.predict(x_test_flat[idx:idx+1], verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit]
    true_digit = y_test[idx]
    
    # Farbe basierend auf Richtig/Falsch
    color = 'green' if predicted_digit == true_digit else 'red'
    
    # Bild anzeigen
    ax = axes[i // 5, i % 5]
    ax.imshow(x_test[idx], cmap='gray')
    ax.set_title(f'Vorher: {predicted_digit} ({confidence:.1%})\nEcht: {true_digit}', 
                color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('10 Zufällige Vorhersagen auf Testset', fontsize=14)
plt.tight_layout()
plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## 8. Modell speichern

# %%
# Modell speichern
model_path = 'mnist_model.h5'
model.save(model_path)
print(f"✓ Modell gespeichert: {model_path}")

# Später wieder laden und testen
loaded_model = keras.models.load_model(model_path)
loaded_test_acc = loaded_model.evaluate(x_test_flat, y_test_encoded, verbose=0)[1]
print(f"✓ Modell geladen und getestet: {loaded_test_acc:.4f} ({loaded_test_acc*100:.2f}%)")



input("Press enter to continue ...")
# This cell just defined all functions needed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import pandas as pd
import matplotlib.pyplot as plt



# ============================================================================ #

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
    model.compile( optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ============================================================================ #

def train(model, X, y, epoche=15, batch_size=32, validation_split=0.2, verbose=1):
    # Training mit Validierungsdaten
    print("Training startet...\n")
    start_time = time.time()
    history = model.fit( X, y, epochs=epoche, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
    training_time = time.time() - start_time
    print(f"\n✓ Training beendet in {training_time:.2f} Sekunden")
    return history, training_time


# ============================================================================ #

def visualize(history):
    # Trainings-Verlauf plotten
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Loss-Plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validierung Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss über Zeit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy-Plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validierung Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Genauigkeit über Zeit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Letzte Werte anzeigen
    print(f"Finaler Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Finaler Validierung Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Finaler Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Finaler Validierung Accuracy: {history.history['val_accuracy'][-1]:.4f}")


# ============================================================================ #

def test(model, X, y, verobse=0):
# Testset evaluieren
    test_loss, test_accuracy = model.evaluate(X, y, verbose=verobse)

    print("\n" + "="*50)
    print("TESTSET ERGEBNISSE")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*50)
    return test_loss, test_accuracy


# ============================================================================ #

def save_model(model, model_path='mnist_model.h5'):
    # Modell speichern
    model.save(model_path)
    print(f"✓ Modell gespeichert: {model_path}")


# ============================================================================ #

def load_model(model, model_path='mnist_model.h5', X=None, y=None):
    # Später wieder laden und testen
    loaded_model = keras.models.load_model(model_path)
    if X and y:
        loaded_test_acc = loaded_model.evaluate(X, y)[1]
        print(f"✓ Modell geladen und getestet: {loaded_test_acc:.4f} ({loaded_test_acc*100:.2f}%)")
    return load_model


# ============================================================================ #

def create_result_entry( name, history, training_time, test_loss, test_accuracy):
    return { "name": name, "history": history, "training_time": training_time, "test_loss": test_loss, "test_accuracy": test_accuracy }


# ============================================================================ #

def results_table(results):
    data = []
    for r in results:
        data.append({ "Modell": r["name"], "Test Accuracy": r["test_accuracy"], "Test Loss": r["test_loss"], "Trainingszeit (s)": r["training_time"] })
    return pd.DataFrame(data)


# ============================================================================ #

def visualize_experiment_results(results):
    names = [ r["name"] for r in results ]
    accuracies = [ r["test_accuracy"] for r in results ]
    times = [ r["training_time"] for r in results ]
    _, axes = plt.subplots( 1, 2, figsize=(12,4))
    # Accuracy
    axes[0].bar( names, accuracies)
    axes[0].set_ylim( 0, 1)
    axes[0].set_ylabel( "Test Accuracy")
    axes[0].set_title( "Vergleich der Genauigkeit")
    # Training time
    axes[1].bar( names, times)
    axes[1].set_ylabel( "Sekunden")
    axes[1].set_title( "Trainingszeit")
    plt.tight_layout()
    plt.show()


# ============================================================================ #

def visualize_learning_curves(results):
    plt.figure(figsize=(8,5))
    for r in results:
        history = r["history"]
        plt.plot( history.history["accuracy"], label=r["name"])
    plt.xlabel( "Epoche")
    plt.ylabel( "Training Accuracy")
    plt.title( "Lernkurven vergleichen")
    plt.grid()
    plt.legend()
    plt.show()

# ============================================================================ #

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

# ============================================================================ #

activations = [ "relu", "sigmoid", "tanh" ]
results = []
for activation in activations:
    model = build_model( hidden_layers=[128,64], activation=activation)
    model.summary()
    history, training_time = train( model, x_train_flat, y_train, epoche=15)
    test_loss, test_accuracy = test( model, x_test_flat, y_test)
    results.append( create_result_entry( name=activation, history=history, training_time=training_time, test_loss=test_loss, test_accuracy=test_accuracy))
results_table(results)
visualize_experiment_results(results)
visualize_learning_curves(results)

# ============================================================================ #

architectures = { "klein":[32], "mittel":[128,64], "tief":[512,256,128] }
results_layers = []
for name, architecture in architectures.items():
    model = build_model( hidden_layers=architecture)
    history, training_time = train( model, x_train_flat, y_train, epoche=5)
    loss, accuracy = test( model, x_test_flat, y_test)
    results_layers.append( create_result_entry( name, history, training_time, loss, accuracy))
results_table(results_layers)
visualize_experiment_results(results_layers)
visualize_learning_curves(results_layers)

# ============================================================================ #

dropouts = [ 0, 0.1, 0.3, 0.5 ]
results_dropout = []
for d in dropouts:
    model = build_model( hidden_layers=[128,64], dropout_rate=d)
    history, training_time = train( model, x_train_flat, y_train, epoche=5)
    loss, accuracy = test( model, x_test_flat, y_test)
    results_dropout.append( create_result_entry( f"Dropout={d}", history, training_time, loss, accuracy))
results_table(results_dropout)
visualize_experiment_results(results_dropout)
visualize_learning_curves(results_dropout)

# ============================================================================ #

learnrates = [ .0001, 0.001, 0.01, 0.1 ]
results_learnrate = []
for rate in learnrates:
    model = build_model( hidden_layers=[128,64], learning_rate=rate)
    history, training_time = train( model, x_train_flat, y_train, epoche=5)
    loss, accuracy = test( model, x_test_flat, y_test)
    results_learnrate.append( create_result_entry( f"Rate={rate}", history, training_time, loss, accuracy))
results_table(results_learnrate)
visualize_experiment_results(results_learnrate)
visualize_learning_curves(results_learnrate)
# %% [markdown]
# # 1. Teil: Bilder verstehen
# ## Wie speichert der Computer Bilder?
# 
# In diesem Notebook werden wir lernen, wie digitale Bilder gespeichert sind und mit dem MNIST-Datensatz arbeiten.

# %%
# import all needed libraries
# python3.14 does not support tensorfow: use different backend
import os
try:
    from tensorflow import keras
except Exception as e:
    print("Cannot import tensorflow: using jax")
    os.environ["KERAS_BACKEND"] = "jax"
    import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# %%
# Ein einfaches 4x4 Schachbrett-Muster als Zahlen
schachbrett = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

print("Das Schachbrett als Zahlen:")
print(schachbrett)

# Das Bild anzeigen
plt.figure(figsize=(4, 4))
plt.imshow(schachbrett, cmap='gray')
plt.title('Schachbrett-Muster (4x4 Pixel)')
plt.colorbar()
plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## 1. Ein einfaches Schwarzweiß-Bild

# %% [markdown]
# ## 2. Graustufenbilder (0-255)

# %%
# Ein Graustufenbild - Werte von 0 (schwarz) bis 255 (weiß)
graustufen = np.array([
    [0, 64, 128, 192],
    [32, 96, 160, 224],
    [64, 128, 192, 255],
    [96, 160, 224, 255]
])

print("Graustufenbild (0-255):")
print(graustufen)

# Anzeigen
plt.figure(figsize=(6, 4))
plt.imshow(graustufen, cmap='gray', vmin=0, vmax=255)
plt.title('Graustufenbild (4x4 Pixel)')
plt.colorbar(label='Helligkeit')
plt.pause(0.3)
plt.show(block=False)

print(f"\nMinimum: {graustufen.min()} (schwarz)")
print(f"Maximum: {graustufen.max()} (weiß)")
print(f"Durchschnitt: {graustufen.mean():.1f}")

# %% [markdown]
# ## 3. Farbbilder (RGB)

# %%
# RGB-Bild: Jedes Pixel hat 3 Werte (Rot, Grün, Blau)
# Shape: (Höhe, Breite, 3 Kanäle)

rgb_bild = np.zeros((4, 4, 3), dtype=np.uint8)

# Ein rotes Quadrat oben links
rgb_bild[0:2, 0:2] = [255, 0, 0]  # Rot

# Ein grünes Quadrat oben rechts
rgb_bild[0:2, 2:4] = [0, 255, 0]  # Grün

# Ein blaues Quadrat unten links
rgb_bild[2:4, 0:2] = [0, 0, 255]  # Blau

# Ein gelbes Quadrat unten rechts
rgb_bild[2:4, 2:4] = [255, 255, 0]  # Gelb

print("RGB-Bild Form:", rgb_bild.shape)
print("Pixel [0,0]:", rgb_bild[0, 0], "(Rot)")
print("Pixel [0,2]:", rgb_bild[0, 2], "(Grün)")
print("Pixel [2,0]:", rgb_bild[2, 0], "(Blau)")
print("Pixel [2,2]:", rgb_bild[2, 2], "(Gelb)")

# Anzeigen
plt.figure(figsize=(6, 6))
plt.imshow(rgb_bild)
plt.title('Einfaches RGB-Bild (4x4 Pixel)')
plt.axis('off')
plt.pause(0.3)
plt.show(block=False)

# %%
import numpy as np
import matplotlib.pyplot as plt

# 16x16 RGB image
size = 16
rgb_bild = np.zeros((size, size, 3), dtype=np.uint8)

for y in range(size):
    for x in range(size):
        r = int(255 * x / (size - 1))                 # Red increases left -> right
        g = int(255 * y / (size - 1))                 # Green increases top -> bottom
        b = int(255 * (x / 4 + y / 2) / (2 * (size - 1)))     # Blue changes smoothly

        rgb_bild[y, x] = [r, g, b]

plt.figure(figsize=(8, 8))
plt.imshow(rgb_bild, interpolation="nearest")
plt.title("16×16 RGB-Bild mit fließenden Farbübergängen")
plt.xticks([])
plt.yticks([])
plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## 4. MNIST-Daten laden und erkunden

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("MNIST Datensatz Informationen:")
print(f"Training Bilder: {x_train.shape}")
print(f"Training Labels: {y_train.shape}")
print(f"Test Bilder: {x_test.shape}")
print(f"Test Labels: {y_test.shape}")

print(f"\nPixel-Werte: min={x_train.min()}, max={x_train.max()}")
print(f"Erste 20 Labels: {y_train[:20]}")

# %% [markdown]
# ## 5. Ein einzelnes MNIST-Bild analysieren

# %%
# Ein einzelnes Bild anschauen
index = 0
bild = x_train[index]
label = y_train[index]

print(f"Bild Index: {index}")
print(f"Label (echte Ziffer): {label}")
print(f"Bild-Form: {bild.shape}")
print(f"\nDie ersten 10 Pixel-Werte der ersten Zeile:")
print(bild[0, :10])

# Das ganze Bild anzeigen
plt.figure(figsize=(6, 6))
plt.imshow(bild, cmap='gray')
plt.title(f'MNIST Bild: Ziffer {label}')
plt.colorbar(label='Helligkeit')
plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## 6. Mehrere MNIST-Bilder vergleichen

# %%
# Mehrere Ziffern einer Klasse anschauen (z.B. alle 3er)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Finde die ersten 10 Ziffern "3"
indices_3 = np.where(y_train == 3)[0][:10]

for i, idx in enumerate(indices_3):
    ax = axes[i // 5, i % 5]
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Ziffer: 3')
    ax.axis('off')

plt.suptitle('Verschiedene Schreibweisen der Ziffer 3', fontsize=14)
plt.tight_layout()
plt.pause(0.3)
plt.show(block=False)

print(f"Beobachtung: Alle sind Ziffern 3, aber die Schreibweise ist unterschiedlich!")

# %% [markdown]
# ## 7. Bild-Statistiken

# %%
# Statistiken / Merkmale für verschiedene Ziffern
print("Zusammengefasste Merkmale verschiedener Ziffern")
print("="*60)

threshold = 100

metrics = []
center_variances = []

for digit in range(10):
    indices = np.where(y_train == digit)[0]
    bilder = x_train[indices]

    # Durchschnittliche Helligkeit
    durchschnitt = bilder.mean()

    # Standardabweichung der Pixelwerte
    std = bilder.std()

    # Anzahl dunkler Pixel
    dunkle_pixel = (bilder < threshold).sum(axis=(1,2)).mean()

    # Schwerpunkt für jedes einzelne Bild
    y_coords, x_coords = np.indices((28, 28))

    gewicht = bilder.sum(axis=(1,2))

    center_x = (
        (bilder * x_coords).sum(axis=(1,2)) / gewicht
    )

    center_y = (
        (bilder * y_coords).sum(axis=(1,2)) / gewicht
    )

    # Mittelwerte
    mean_x = center_x.mean()
    mean_y = center_y.mean()

    # Standardabweichung der Positionen
    std_x = center_x.std()
    std_y = center_y.std()

    metrics.append([
        durchschnitt,
        std,
        dunkle_pixel,
        mean_x,
        mean_y
    ])

    center_variances.append([
        std_x,
        std_y
    ])

    print(
        f"Ziffer {digit}: "
        f"Schwerpunkt=({mean_x:.1f},{mean_y:.1f}), "
        f"Streuung=({std_x:.1f},{std_y:.1f})"
    )


metrics = np.array(metrics)
center_variances = np.array(center_variances)

# %%
# -------------------------------------------------
# Visualisierung
# -------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 8))


# Durchschnittliche Helligkeit
axes[0].bar(range(10), metrics[:,0])
axes[0].set_title("Durchschnittliche Helligkeit")
axes[0].set_xlabel("Ziffer")
axes[0].set_ylabel("Helligkeit")


# Anzahl dunkler Pixel
axes[1].bar(range(10), metrics[:,2])
axes[1].set_title("Anzahl dunkler Pixel")
axes[1].set_xlabel("Ziffer")
axes[1].set_ylabel("Pixel")


# -------------------------------------------------
# Visualisierung: Schwerpunkt als 2D Feature Space
# -------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    metrics[:, 3],   # horizontaler Schwerpunkt
    metrics[:, 4],   # vertikaler Schwerpunkt
)

for digit in range(10):
    plt.annotate(
        str(digit),
        (metrics[digit, 3], metrics[digit, 4]),
        fontsize=12,
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Horizontaler Schwerpunkt (x)")
plt.ylabel("Vertikaler Schwerpunkt (y)")
plt.title("Ziffern im Merkmalsraum: Schwerpunkt der Pixel")

plt.grid(True)
plt.pause(0.3)
plt.show(block=False)


plt.suptitle(
    "Zusammengefasste Merkmale verschiedener Ziffern",
    fontsize=14
)

plt.tight_layout()
plt.pause(0.3)
plt.show(block=False)

# %%
# -------------------------------------------------
# Visualisierung: Schwerpunkt mit Streuung
# -------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    metrics[:, 3],
    metrics[:, 4],
)

ax = plt.gca()

for digit in range(10):

    x = metrics[digit, 3]
    y = metrics[digit, 4]

    # Größe der Ellipse aus Standardabweichungen
    width = 2 * center_variances[digit, 0]
    height = 2 * center_variances[digit, 1]

    ellipse = plt.Circle(
        (x, y),
        radius=max(width, height),
        fill=False
    )

    ax.add_patch(ellipse)

    plt.annotate(
        str(digit),
        (x, y),
        fontsize=12,
        xytext=(5,5),
        textcoords="offset points"
    )


plt.xlabel("Horizontaler Schwerpunkt (x)")
plt.ylabel("Vertikaler Schwerpunkt (y)")
plt.title(
    "Ziffern im Merkmalsraum mit Streuung der Handschriften"
)

plt.grid(True)
plt.axis("equal")

plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## 8. Durchschnittliche Ziffern visualisieren

# %%
# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert to float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# --------------------------------------------------
# Compute average image of every digit
# --------------------------------------------------

average_digits = np.zeros((10, 28, 28))

for digit in range(10):
    average_digits[digit] = x_train[y_train == digit].mean(axis=0)

# --------------------------------------------------
# Visualize
# --------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(12,5))

for digit in range(10):
    ax = axes[digit//5, digit%5]
    ax.imshow(average_digits[digit], cmap="gray")
    ax.set_title(str(digit))
    ax.axis("off")

plt.suptitle("Average digit")
plt.pause(0.3)
plt.show(block=False)

# %%
def predict(image, prototypes):
    # Euclidean distance
    distances = np.linalg.norm(prototypes - image, axis=(1,2))

    # Smaller distance -> larger probability
    scores = np.exp(-distances / 1000)

    probabilities = scores / scores.sum()

    prediction = np.argmax(probabilities)

    return prediction, probabilities

# %%
def test_number(idx):
    prediction, probs = predict(x_test[idx], average_digits)
    
    plt.imshow(256 - x_test[idx], cmap="gray")
    plt.title(f"Prediction: {prediction}   True: {y_test[idx]}")
    plt.axis("off")
    plt.pause(0.3)
    plt.show(block=False)
    
    print(probs)
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), probs)
    plt.xlabel('Ziffer')
    plt.ylabel('Probability')
    plt.title('Prediction Probability')
    plt.xticks(range(10))
    plt.pause(0.3)
    plt.show(block=False)

# %%
test_number(1)

# %%
test_number(1000)

# %%
predictions = []

for image in x_test:
    pred, _ = predict(image, average_digits)
    predictions.append(pred)

predictions = np.array(predictions)

accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy:.3f}")

# %%
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.pause(0.3)
plt.show(block=False)

# %% [markdown]
# ## Ein lernender Klassifikator: k-Nächste-Nachbarn

# %%
# Flatten images
X_train = x_train.reshape(len(x_train), -1)
X_test = x_test.reshape(len(x_test), -1)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")

# %%
def test_number_kNN(idx):
    distances, indices = knn.kneighbors([X_test[idx]], n_neighbors=5)
    print(distances) 
    plt.figure(figsize=(12,3))
    
    plt.subplot(1,6,1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"Test\n{y_test[idx]}")
    plt.axis("off")
    
    for i, ind in enumerate(indices[0]):
        plt.subplot(1,6,i+2)
        plt.imshow(x_train[ind], cmap="gray")
        plt.title(y_train[ind])
        # plt.axis("off")
        plt.xlabel("d = " + str(int(distances[0][i])))
    
    plt.tight_layout()
    plt.pause(0.3)
    plt.show(block=False)

# %%
test_number_kNN(0)

# %%
test_number_kNN(1234)

# %%
def nkk_confusionmatrix():
    predictions = []

    def predict_digit(image):
        image = image.reshape(1, -1)
        return knn.predict(image)[0]

    predictions = np.array([predict_digit(img) for img in x_test])

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.pause(0.3)
    plt.show(block=False)

# %%
# nkk_confusionmatrix()


input("Press enter to continue ...")
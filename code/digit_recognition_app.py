#!/usr/bin/env python3
"""
Handschriften-Ziffer Erkennungsanwendung
Web-basierte GUI für die Erkennung von handgeschriebenen Ziffern mit Deep Learning
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import sys

# Modell-Pfad
MODEL_PATH_0 = 'final_mnist_model.h5'
MODEL_PATH_1 = 'mnist_model.h5'

app = Flask(__name__)

# Globale Variable für das Modell
model = None

def center_image(img):

    coords = np.argwhere(img > 0.1)

    if len(coords) == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[
        y_min:y_max+1,
        x_min:x_max+1
    ]

    return cropped

app = Flask(__name__)

# Globale Variable für das Modell
model = None

def load_model():
    """Lade das trainierte Modell"""
    global model
    for MODEL_PATH in [MODEL_PATH_0, MODEL_PATH_1]:
        if os.path.exists(MODEL_PATH):
            try:
                model = keras.models.load_model(MODEL_PATH)
                print(f"✓ Modell erfolgreich geladen: {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"✗ Fehler beim Laden des Modells: {e}")
                model = None
        else:
            print(f"✗ Modell nicht gefunden: {MODEL_PATH}")
            print("  Bitte trainieren Sie zuerst das Modell mit den Notebooks!")
            print("  Folgen Sie: code/05_Erstes_Netzwerk.ipynb")
            model = None
    return False

@app.route('/')
def index():
    """Hauptseite - Anwendungsinterface"""
    return render_template('digit_app.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API-Endpunkt für Vorhersagen
    Erwartet JSON mit 'image' (Canvas als PNG in Base64)
    """
    
    if model is None:
        return jsonify({
            'error': 'Modell nicht geladen. Bitte erst das Modell trainieren!'
        }), 400
    
    try:
        # Bilddaten aus Request extrahieren
        data = request.json
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'error': 'Keine Bilddaten erhalten'}), 400
        
        # Base64 dekodieren
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Graustufen
        image = image.convert('L')

        # numpy array
        img_array = np.array( image, dtype=np.float32) / 255.0
        # Invertieren:
        # Hintergrund 0, Zahl 1
        img_array = 1 - img_array
        # Zentrieren
        # img_array = center_image(img_array)
        # wieder auf 28x28 bringen
        image = Image.fromarray( (img_array * 255).astype(np.uint8))
        image = image.resize( (28,28), Image.Resampling.LANCZOS)
        # zurück zu float

        # 28x28 Bild für Anzeige erzeugen
        small_image = image.resize((280, 280), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        small_image.save(buffer, format="PNG")
        small_image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        img_array = np.array( image, dtype=np.float32) / 255.0

        # try CNN input shape
        try: 
            img_array = img_array.reshape( 1, 28, 28, 1)
            prediction = model.predict( img_array, verbose=0)
        except: 
            img_array = img_array.reshape(1, 784)  # Flatten to (1, 784) for Dense layer
            prediction = model.predict( img_array, verbose=0)

        digit = int( np.argmax(prediction[0]))
        confidence = float(prediction[0][digit] * 100)
        
        # Wahrscheinlichkeiten für alle Ziffern
        probabilities = { str(i): float(prediction[0][i] * 100) for i in range(10) }
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': confidence,
            'probabilities': probabilities,
            'processed_image': 'data:image/png;base64,' + small_image_base64
        })
    
    except Exception as e:
        print(f"Fehler in predict(): {e}")
        return jsonify({'error': f'Fehler bei der Vorhersage: {str(e)}'}), 400

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Prüfe, ob das Modell geladen ist"""
    return jsonify({
        'loaded': model is not None,
        'model_path': MODEL_PATH_0
    })

if __name__ == '__main__':
    print()
    print("=" * 70)
    print("🤖 Deep Learning - Handschriften-Ziffer Erkennung")
    print("=" * 70)
    print()
    
    # Versuche das Modell zu laden
    model_loaded = load_model()
    
    print()
    print("Web-Anwendung wird gestartet...")
    print()
    print("📱 Öffne deinen Browser und gehe zu:")
    print()
    print("   👉  http://localhost:5000")
    print()
    print("Dann:")
    print("   1️⃣  Zeichne eine Ziffer (0-9) ins weiße Feld")
    print("   2️⃣  Klicke 'Vorhersage machen'")
    print("   3️⃣  Das Modell zeigt dir, welche Ziffer es erkannt hat!")
    print()
    print("Zum Beenden drücke: Ctrl+C")
    print("=" * 70)
    print()
    
    # Starte den Flask-Server
    try:
        app.run(debug=False, port=5000, host='localhost')
    except KeyboardInterrupt:
        print("\n\nAnwendung beendet.")
        sys.exit(0)

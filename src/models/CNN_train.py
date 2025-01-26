import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from module_for_preprocessing import *
from tensorflow.keras import layers, models
import yaml
from datetime import datetime  # Nécessaire pour générer un timestamp unique

# Charger la configuration
with open("../../config.yml", "r") as file:
    config = yaml.safe_load(file)
size = config["données"]["image"]["size"]

# Chargement des datasets
training = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Training"
)
validation = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Validation"
)
testing = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Testing"
)

x_train, y_train = load_images_with_preprocessing(training, size)
x_val, y_val = load_images_with_preprocessing(validation, size)
x_test, y_test = load_images_with_preprocessing(testing, size)

# Définir le modèle CNN
model = models.Sequential(
    [
        layers.Input(shape=(256, 256, 1)),  # (256, 256, 3) si on veut RGB
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        # Bloc 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        # Bloc 4
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        # Couche Fully Connected
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),  # Pour éviter le surapprentissage
        layers.Dense(4, activation="softmax"),  # Multiclass classification
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Résumé du modèle
model.summary()

# Entraînement
history = model.fit(
    x_train, y_train, batch_size=32, epochs=6, validation_data=(x_val, y_val)
)

# Évaluation du modèle sur le jeu de test
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Sauvegarder le modèle
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, "sauvegardes_modeles")
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(save_dir, f"modele_brain_tumor_{timestamp}.h5")
model.save(model_path)
print(f"Modèle sauvegardé dans : {model_path}")

# Sauvegarder l'historique
history_dict = {
    f"model_{timestamp}": {
        "loss": history.history["loss"],
        "accuracy": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
}

# Chemin du fichier JSON
history_file = os.path.join(save_dir, "models_history.json")

# Charger ou créer le fichier JSON
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        all_histories = json.load(f)
else:
    all_histories = {}

# Ajouter les nouvelles données
all_histories.update(history_dict)

# Sauvegarder le fichier JSON mis à jour
with open(history_file, "w") as f:
    json.dump(all_histories, f, indent=4)

print(f"Historique sauvegardé dans : {history_file}")

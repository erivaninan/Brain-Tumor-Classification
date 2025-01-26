import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import streamlit as st
import yaml
import json
from datetime import datetime  # Ajout de l'import
from module_for_preprocessing import load_images_with_preprocessing

# Charger la configuration
with open("../../config.yml", "r") as file:
    config = yaml.safe_load(file)
size = config["données"]["image"]["size"]


# Chargement des datasets
base_data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
training = os.path.join(base_data_path, "data", "Training")
validation = os.path.join(base_data_path, "data", "Validation")
testing = os.path.join(base_data_path, "data", "Testing")

# Prétraitement des images
x_train, y_train = load_images_with_preprocessing(training, size)
x_val, y_val = load_images_with_preprocessing(validation, size)
x_test, y_test = load_images_with_preprocessing(testing, size)

# Charger le modèle sauvegardé
@st.cache_resource
def load_cnn_model():
    save_dir = os.path.join(os.getcwd(), "sauvegardes_modeles")
    model_to_load = "modele_brain_tumor_20241214_145523.h5" #model a entrainer
    model_path = os.path.abspath(os.path.join(save_dir, model_to_load))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle spécifié n'existe pas : {model_path}")
    
    model = load_model(model_path)
    st.write(f"Modèle chargé : {model_path}")
    st.write(f"Dimensions d'entrée attendues : {model.input_shape}")
    return model

model = load_cnn_model()


# Continuer l'entraînement
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,  # Nombre d'époques supplémentaires
    batch_size=32,
)

# Évaluation du modèle sur le jeu de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Sauvegarder le modèle
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, "sauvegardes_modeles")
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Utilisation de datetime maintenant disponible
model_path = os.path.join(save_dir, f"modele_brain_tumor_{timestamp}_finetuning.h5")
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

# Chemin du fichier JSON pour l'historique
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

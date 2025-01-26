import os
import numpy as np
import tensorflow as tf
from module_for_preprocessing_metrics import (
    load_saved_model,
    load_images_with_preprocessing,
    courbe_ROC,
    get_test_loss,
    get_test_accuracy,
    plot_confusion_matrix,
)
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import streamlit as st

CLASS_NAMES = {
    0: "Gliome",
    1: "Méningiome",
    2: "Pas de tumeur",
    3: "Pituitary",
}

# -----------------------------------------
# Charger le modèle

model_path = "modele_brain_tumor_20241117_205951.h5"
model = load_saved_model(model_path)


# -----------------------------------------
# Fonction pour prédire la classe d'une image
def predict_image_class(image, model, target_size=(256, 256)):
    """
    Prédit la classe d'une image donnée à l'aide d'un modèle CNN.
    """

    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0  # Normaliser 
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CLASS_NAMES.get(predicted_class, "Classe inconnue")


# -----------------------------------------

testing_dir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Testing"
)
x_test, y_test = load_images_with_preprocessing(testing_dir, size=256)

# Charger l'historique des modèles
models_history_path = os.path.join("sauvegardes_modeles", "models_history.json")


# -----------------------------------------
# Partie Streamlit

# Titre de l'application
st.title("Classification d'image : Tumeurs Cérébrales")

# Texte explicatif
st.write(
    """
Suite au benchmarking des modèles de CNN, nous avons décidé d’ajouter une dizaine d’époques au meilleur modèle pour obtenir de meilleures performances et vous présenter cette application Streamlit.

Voici les nouvelles performances du modèle :
"""
)

# Afficher les performances
pictures_dir = os.path.join(
    os.getcwd(), "pictures"
)  

# Afficher la courbe ROC
roc_curve_path = os.path.join(pictures_dir, "courbe_ROC.png")
if os.path.exists(roc_curve_path):
    st.write("### Courbe ROC")
    st.image(roc_curve_path, caption="Courbe ROC", use_column_width=True)

# Afficher la matrice de confusion
confusion_matrix_path = os.path.join(pictures_dir, "confusion_matrix.png")
if os.path.exists(confusion_matrix_path):
    st.write("### Matrice de Confusion")
    st.image(
        confusion_matrix_path, caption="Matrice de Confusion", use_column_width=True
    )

st.write("### Test Loss")
st.write("Test Loss : 4.393250942230225")

st.write("### Test Accuracy")
st.write("Test Accuracy : 0.7182741165161133")

# Option de téléchargement de fichier
st.write(
    """

Enfin, voici un **sample de 2 images** à télécharger. Testez l’application pour découvrir si elles contiennent une tumeur (ou pas) !
"""
)

# Construire les chemins des images dans le répertoire 'samples'
samples_dir = os.path.join(
    os.getcwd(), "samples"
)  
sample_images = [
    os.path.join(samples_dir, "image(1).jpg"),
    os.path.join(samples_dir, "image(2).jpg"),
]

# Afficher les images avec des boutons de téléchargement
st.write("### Exemple d'images à tester")
st.write("Cliquez sur une image pour la télécharger.")
cols = st.columns(2)  

for i, image_path in enumerate(sample_images):
    if os.path.exists(image_path):
        # Charger l'image
        with open(image_path, "rb") as file:
            image_data = file.read()

        # Afficher l'image et ajouter un bouton de téléchargement en dessous
        with cols[i % 2]:  
            st.image(image_path, use_column_width=True, caption=f"Image {i+1}")
            st.download_button(
                label="Télécharger",
                data=image_data,
                file_name=os.path.basename(image_path),
                mime="image/jpeg",
            )


uploaded_file = st.file_uploader(
    "Choisissez une image (JPG/PNG)...", type=["jpg", "jpeg", "png"]
)

# Prédire la classe d'une image téléchargée
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Image téléchargée", use_column_width=True)


    st.write("Classification en cours...")
    predicted_class = predict_image_class(image, model)


    st.write(f"Classe prédite : **{predicted_class}**")

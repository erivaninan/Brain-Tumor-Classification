# _________________________________________
# Importations

import os
import yaml
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from PIL import Image
import plotly.subplots as sp
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# _________________________________________
# Chargement de la configuration

with open("../config.yml", "r") as file:
    config = yaml.safe_load(file)

image_size = config["données"]["image"]["size"]
normalize = config["données"]["image"]["normalize"]
standardize = config["données"]["image"]["standardize"]
grayscale = config["données"]["image"]["grayscale"]

# _________________________________________
# Fonctions de prétraitement des images


def resize_and_rename_images_of_folder(folder, subfolder, size):
    """
    Redimensionne et renomme les images dans un dossier donné.
    """
    data_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", folder, subfolder
    )
    resized = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), "../..")),
        "data",
        folder,
        f"resized_{subfolder}_{size}",
    )
    os.makedirs(resized, exist_ok=True)

    for count, filename in enumerate(os.listdir(data_path)):
        image_path = os.path.join(data_path, filename)
        with Image.open(image_path) as image:
            image_resized = image.resize((size, size))
            output_path = os.path.join(resized, f"{subfolder}_{count}.jpg")
            image_resized.save(output_path)


def load_and_preprocess_image(image_path, grayscale, normalize, standardize):
    """
    Charge et prétraite une image.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1 if grayscale else 3)

    if normalize:
        img = img / 255.0
    if standardize:
        img = (img - tf.reduce_mean(img)) / tf.math.reduce_std(img)

    return img


def load_images_with_preprocessing(directory, size):
    """
    Charge et prétraite un ensemble d'images depuis un dossier.
    """
    images = []
    labels = []
    subfolders = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    class_names = [f"resized_{subfolder}_{size}" for subfolder in subfolders]

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert("L")
            img_array = np.expand_dims(np.array(img), axis=-1)
            images.append(img_array)
            labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# _________________________________________
# Fonction pour charger un modèle sauvegardé


def load_saved_model(model_name):
    """
    Charge un modèle sauvegardé depuis le répertoire des modèles.
    """
    save_dir = os.path.join("..", "src", "models", "sauvegardes_modeles")
    model_path = os.path.join(save_dir, model_name)
    return load_model(model_path)


# _________________________________________
# Fonctions de visualisation et d'évaluation


def plot_confusion_matrix(model, x_test, y_test):
    """
    Affiche la matrice de confusion pour un modèle donné.
    """
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix",
    )
    fig.update_layout(xaxis_title="Predicted Labels", yaxis_title="True Labels")
    fig.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))


def courbe_ROC(model, x_test, y_test):
    """
    Affiche les courbes ROC pour un modèle donné.
    """
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    y_pred_proba = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    n_classes = y_pred_proba.shape[1]
    y_true_onehot = label_binarize(y_true, classes=range(n_classes))

    roc_data = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        roc_data.append(
            pd.DataFrame(
                {
                    "FPR": fpr,
                    "TPR": tpr,
                    "Class": f"{class_names[i]} (AUC = {auc_score:.2f})",
                }
            )
        )

    combined_data = pd.concat(roc_data)
    fig = px.line(combined_data, x="FPR", y="TPR", color="Class", title="ROC Curve")
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray")
    )
    fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )
    fig.show()


def plot_model_curves(json_file, model_name):
    """
    Affiche les courbes de perte et de précision pour un modèle spécifique.
    """
    with open(json_file, "r") as file:
        histories = json.load(file)

    if model_name not in histories:
        print(f"Le modèle '{model_name}' n'existe pas dans le fichier JSON.")
        return

    history = histories[model_name]
    epochs = list(range(1, len(history["accuracy"]) + 1))

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["accuracy"],
            mode="lines+markers",
            name="Training Accuracy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_accuracy"],
            mode="lines+markers",
            name="Validation Accuracy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["loss"], mode="lines+markers", name="Training Loss"
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=f"Model Performance Curves: {model_name}",
        xaxis_title="Epochs",
        template="plotly_white",
    )
    fig.show()


# _________________________________________
# Fonctions de récupération des métriques


def get_test_accuracy(json_file, model_name):
    """
    Récupère la précision du test pour un modèle spécifique.
    """
    with open(json_file, "r") as file:
        histories = json.load(file)

    if model_name in histories:
        model_data = histories[model_name]
        return model_data.get("test_accuracy") or model_data.get("metrics", {}).get(
            "test_accuracy"
        )

    print(f"Le modèle '{model_name}' n'existe pas ou ne contient pas de test_accuracy.")
    return None


def get_test_loss(json_file, model_name):
    """
    Récupère la perte du test pour un modèle spécifique.
    """
    with open(json_file, "r") as file:
        histories = json.load(file)

    if model_name in histories:
        model_data = histories[model_name]
        return model_data.get("test_loss") or model_data.get("metrics", {}).get(
            "test_loss"
        )

    print(f"Le modèle '{model_name}' n'existe pas ou ne contient pas de test_loss.")
    return None


def classification_report_aggregated(model, x_test, y_test):
    """
    Agrège les métriques pour calculer les moyennes sur les 4 classes.
    """
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True
    )

    return {
        "precision_avg": report["weighted avg"]["precision"],
        "recall_avg": report["weighted avg"]["recall"],
        "f1_score_avg": report["weighted avg"]["f1-score"],
    }


def plot_accuracy_histogram(accuracy, labels):
    """
    Trace un histogramme simple des précisions des modèles.

    Args:
        accuracy (list): Liste des précisions des modèles.
        labels (list): Liste des noms des modèles correspondants.
    """

    fig = px.bar(
        x=labels,
        y=accuracy,
        title="Test Accuracy",
        labels={"x": "Model", "y": "Accuracy (%)"},
    )


    fig.show()



# Fonction pour prédire la classe d'une image
def predict_image_class(image, model, target_size=(256, 256)):
    """
    Prédit la classe d'une image donnée à l'aide d'un modèle CNN.
    """
    # Prétraiter l'image
    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0  # Normaliser entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Prédire la classe
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class


# Fonction principale pour obtenir les k premières bonnes prédictions
def get_k_good(model, k, dir):
    """
    Charge les images de test, prédit les classes et retourne k échantillons correctement classés.

    """
    CLASS_NAMES = {
    0: "glioma_tumor",
    1: "meningioma_tumor",
    2: "no_tumor",
    3: "pituitary_tumor"
    }

    # Préparer les données de test
    testing_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data", dir))
    correct_predictions_dir = os.path.join(os.getcwd(), "correct_predictions")

    # Créer un répertoire pour sauvegarder les images correctement classées
    if os.path.exists(correct_predictions_dir):
        shutil.rmtree(correct_predictions_dir)
    os.makedirs(correct_predictions_dir)

    # Initialiser X_good et Y_good
    X_good = []
    Y_good = []

    # Parcourir les images de test et trouver les bonnes prédictions
    for class_index, class_name in CLASS_NAMES.items():
        class_dir = os.path.join(testing_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Dossier non trouvé pour la classe {class_name} : {class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    # Charger l'image
                    image = Image.open(file_path).convert("L")

                    # Prédire la classe
                    predicted_class = predict_image_class(image, model)

                    # Vérifier si la prédiction est correcte
                    if predicted_class == class_index:
                        # Ajouter l'image et l'étiquette à X_good et Y_good
                        img_array = img_to_array(image.resize((256, 256))) / 255.0
                        X_good.append(img_array)
                        Y_good.append(class_index)

                        # Sauvegarder l'image correctement classée
                        output_path = os.path.join(
                            correct_predictions_dir,
                            f"{class_name}_{os.path.basename(file_path)}",
                        )
                        image.save(output_path)
                        print(
                            f"Image correctement classée pour {class_name} sauvegardée : {output_path}"
                        )

                        # Arrêter si on a atteint k bonnes prédictions
                        if len(X_good) >= k:
                            X_good = np.array(X_good)
                            Y_good = np.array(Y_good)
                            return X_good, Y_good
                except Exception as e:
                    print(f"Erreur avec l'image {file_path} : {e}")

    # Convertir X_good et Y_good en tableaux NumPy
    X_good = np.array(X_good)
    Y_good = np.array(Y_good)

    # Vérification finale
    if len(X_good) == 0:
        print("Aucune image correctement classée trouvée.")
    else:
        print(f"X_good et Y_good créés avec succès.")
        print(f"X_good contient {len(X_good)} images correctement classées.")
        print(f"Y_good contient les étiquettes correspondantes.")

    return X_good, Y_good

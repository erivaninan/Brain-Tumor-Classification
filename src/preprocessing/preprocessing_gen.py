# %%
import os
import yaml
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from module_for_preprocessing import *  # fonctions pour ce script

# Charger la configuration
with open('../../config.yml', 'r') as file:
    config = yaml.safe_load(file)

size = config['données']['image']['size']
training_sample = config['données']['validation']['training_sample']  # pourcentage pour validation
subfolders = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# %%

# Étape 1 : Redimensionnement et renommage des images
for subfolder in subfolders:
    resize_and_rename_images_of_folder('Training', subfolder, size)
    resize_and_rename_images_of_folder('Testing', subfolder, size)

classes = [f"resized_{subfolder}_{size}" for subfolder in subfolders]

# Dossier d'entraînement et validation
training = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Training")
validation = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Validation")

# %%

# Étape 2 : Déterminer la classe avec le plus d'images (référence)
class_counts = {}
for classe in classes:
    class_path = os.path.join(training, classe)
    class_counts[classe] = len(os.listdir(class_path))

max_images = max(class_counts.values())  # Nombre maximum d'images dans une classe
print(class_counts)
print(f"Classe de référence : {max_images} images")

# %%

# Étape 3 : Augmentation pour équilibrer toutes les classes
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

for classe in classes:
    class_path = os.path.join(training, classe)
    current_count = len(os.listdir(class_path))

    if current_count < max_images:
        images_to_generate = max_images - current_count  # Nombre d'images à générer
        print(f"Augmentation pour {classe} : {images_to_generate} images à générer")

        # Boucle sur les images existantes pour augmenter
        images = os.listdir(class_path)
        for image_name in images:
            img_path = os.path.join(class_path, image_name)

            # Charger l'image
            img = Image.open(img_path).convert('L')  # Convertir en niveaux de gris si nécessaire
            img_array = np.array(img)               # Pas de redimensionnement
            img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], 1))  # Préparer pour le générateur

            # Générer des images augmentées
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=class_path, save_prefix=f"aug_{os.path.splitext(image_name)[0]}", save_format="png"):
                images_to_generate -= 1
                if images_to_generate <= 0:
                    break
            if images_to_generate <= 0:
                break

    print(f"Augmentation terminée pour {classe}. Total d'images : {len(os.listdir(class_path))}")

# %%

# Étape 4 : Création des sous-dossiers de validation et déplacement des images
for classe in classes:
    class_path = os.path.join(training, classe)
    images = os.listdir(class_path)

    # Sélection aléatoire pour validation
    validation_images = random.sample(images, int(len(images) * training_sample))
    validation_classe = os.path.join(validation, f'{classe}')
    os.makedirs(validation_classe, exist_ok=True)

    for image in validation_images:
        train_path = os.path.join(class_path, image)
        val_path = os.path.join(validation_classe, image)
        os.rename(train_path, val_path)  # Déplacement des images vers validation

# %%

# Étape 5 : Vérification des répartitions
training_counts = {}
validation_counts = {}

for classe in classes:
    class_path = os.path.join(training, classe)
    training_counts[classe] = len(os.listdir(class_path))
    validation_classe = f'{classe}'
    validation_path = os.path.join(validation, validation_classe)
    validation_counts[validation_classe] = len(os.listdir(validation_path))

print("Nombre d'images dans chaque classe d'entraînement :")
for classe, count in training_counts.items():
    print(f"{classe} : {count} images")

print("\nNombre d'images dans chaque classe de validation :")
for classe, count in validation_counts.items():
    print(f"{classe} : {count} images")

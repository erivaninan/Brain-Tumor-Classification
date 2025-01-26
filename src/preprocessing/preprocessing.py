# %%
import os
import yaml
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from module_for_preprocessing import *  # fonctions pour ce script

# Charger la configuration
with open('../../config.yml', 'r') as file:
    config = yaml.safe_load(file)

size = config['données']['image']['size']
training_sample = config['données']['validation']['training_sample']  # pourcentage pour validation
subfolders = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# %%

for subfolder in subfolders:
    resize_and_rename_images_of_folder('Training', subfolder, size)
    resize_and_rename_images_of_folder('Testing', subfolder, size)

classes = [f"resized_{subfolder}_{size}" for subfolder in subfolders]

# %%
training = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Training")
validation = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Validation")

# %%
# Pouvoir run ce script plusieurs fois sans probleme
if os.path.exists(validation):
    for root, dirs, files in os.walk(validation, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(validation)

os.makedirs(validation, exist_ok=True)

# Créer les sous-dossiers de validation et déplacer les images
for classe in classes:
    class_path = os.path.join(training, classe)
    images = os.listdir(class_path)

    validation_images = random.sample(images, int(len(images) * training_sample))  # sélection aléatoire pour la validation
    validation_classe = os.path.join(validation, f'{classe}')
    os.makedirs(validation_classe, exist_ok=True)

    for image in validation_images:
        train_path = os.path.join(class_path, image)
        val_path = os.path.join(validation_classe, image)
        os.rename(train_path, val_path)  # Enlever l'image du training et la mettre dans validation (plus simple)

# %%

# dans ce bout de code on verifie que on a une bonne repartition
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



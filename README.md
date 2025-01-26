# Détection et Classification des Tumeurs Cérébrales

README rédigé par Lilya-Nada KHELID @Lilyakhelid

## Aperçu 🌟

Ce projet a été réalisé dans le cadre d'un cours d'apprentissage statistique. Il s'agit d'un premier projet en traitement d'image.

Ce projet est conçu pour détecter et classifier les tumeurs cérébrales à l'aide d'images IRM. Il prend en charge quatre classes :

- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**
- **Pituitary Tumor**

Le projet inclut des fonctionnalités pour :

- L'exploration des données
- L'évaluation des métriques du modèle
- Les analyses explicatives (explainability)
- Une application Streamlit pour tester le modèle

---

## Jeu de Données 📊

Le jeu de données utilisé pour ce projet est disponible sur Kaggle : [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data).

### Résumé des Données

- **IRM** : Classées dans les quatre catégories mentionnées ci-dessus.
- **Structure des dossiers** : Dossiers séparés pour l'entraînement, la validation et les tests.
- **Prétraitement** : Les images sont redimensionnées, normalisées et converties en niveaux de gris pour l'entraînement du modèle.

---

## Fonctionnalités 🛠️

### 1. Exploration des Données 🔍

Un rapport HTML interactif fournit une vue d'ensemble complète du jeu de données.

### 2. Métriques du Modèle 📈

Un rapport HTML détaillé présentant les métriques d'évaluation telles que la précision, le rappel, et le F1-score.

### 3. Explicabilité 🧠

Des analyses visuelles des décisions du modèle à l'aide de [Xplique](https://github.com/deel-ai/xplique).

### 4. Application Streamlit 🌐

Une interface intuitive permettant de téléverser des images IRM et d'obtenir les résultats de classification en temps réel.
**Note : L'application nécessite le modèle pré-entraîné, qui n'est pas inclus dans ce dépôt GitHub. Vous devez placer le fichier du modèle dans le dossier approprié avant d'utiliser l'application.**

---

## Installation et Exécution ⚙️

### Prérequis

- **Python** : Version 3.11 ou supérieure
- **Configuration système** : Un système capable d'exécuter des environnements virtuels

### Instructions

1. **Cloner le dépôt** :

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Exécuter le script d'installation (Linux/Mac)** :

   ```bash
   ./lancez_moi.sh
   ```

   Ce script effectuera les actions suivantes :

   - Créer un environnement virtuel.
   - Activer l'environnement virtuel.
   - Ouvrir le rapport **exploration des données** dans votre navigateur par défaut.
   - Consulter le rapport **métriques du modèle**.
   - Accéder aux **analyses explicatives**.
   - Lancer l'application **Streamlit**.

3. **Exécuter le script d'installation (Windows)**

   ```bash
   lancez_moi_windows.sh
   ```

   Ce script est conçu pour les utilisateurs Windows.

4. **Lancer l'application Streamlit manuellement** :

   ```bash
   cd src/models
   streamlit run CNN_app.py
   ```

---

## Structure des Dossiers 📂

```plaintext
project-root/
|
|-- data/                 # Dossier contenant les données (entraînement, validation, tests)
|-- Data_visualisation/   # Scripts et fichiers pour la visualisation des données
|-- results/              # Rapports et résultats (par exemple, metrics.html, analyses explicatives)
|-- src/                  # Code source du projet
|   |-- models/           # Scripts pour les modèles, l'entraînement et la détection
|   |   |-- CNN_app.py    # Application Streamlit pour la classification
|   |   |-- CNN_train.py  # Script pour l'entraînement du modèle CNN
|   |   |-- preprocessing/ # Modules de prétraitement des images
|-- venv/                 # Fichiers de l'environnement virtuel
|-- .gitignore            # Règles Git ignore
|-- config.yml            # Fichier de configuration du projet
|-- lancez_moi.sh         # Script d'installation pour Linux/Mac
|-- lancez_moi_windows.sh # Script d'installation pour Windows
|-- README.md             # Documentation du projet
|-- requirements.txt      # Dépendances Python
```

---


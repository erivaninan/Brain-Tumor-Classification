:: Créer un environnement virtuel
python -m venv venv_brain_tumor

:: Activer l'environnement virtuel
venv_brain_tumor\Scripts\activate

:: Installer les dépendances
pip install -r requirements.txt

:: Naviguer vers Data_visualisation
cd Data_visualisation

:: Ouvrir le fichier HTML
start Data_visualisation.html

:: Retour au dossier parent
cd ..

:: Aller dans le dossier results
cd results

:: Ouvrir les fichiers HTML
start metrics.html
start Xplique.html

:: Retour au dossier parent
cd ..

:: Aller dans src/models
cd src\models

:: Lancer l'application Streamlit
streamlit run CNN_app.py

#!/bin/bash


python -m venv venv_brain_tumor

source venv_brain_tumor/bin/activate

pip install -r requirements.txt

cd Data_visualisation

open Data_visualisation.html  #1

cd ..

cd results

open metrics.html  #2

open Xplique.html #3

cd ..

cd src

cd models

#streamlit run CNN_app.py #4

#./lancez_moi.sh

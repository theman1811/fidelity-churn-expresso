import streamlit as st
import joblib
import pandas as pd

import sklearn  # Ajout crucial

sklearn.set_config(transform_output="pandas")  # Nécessaire pour les nouvelles versions

# Chargement du modèle et encodeur
model = joblib.load('logreg_model.pkl')
le_tenure = joblib.load('le_tenure.pkl')

# Configuration de l'interface
st.title('Prédiction de Churn Expresso 📱')
st.markdown("Prédisez si un client risque de se désabonner")

# Section de saisie des caractéristiques
st.header("Caractéristiques du client")

# Récupération des valeurs originales pour TENURE
tenure_options = le_tenure.classes_

# Création des champs de saisie
tenure = st.selectbox('Ancienneté (TENURE)', options=tenure_options)
montant = st.number_input('Montant top-up (MONTANT)', min_value=0.0)
freq_rech = st.number_input('Fréquence rechargement (FREQUENCE_RECH)', min_value=0.0)
arpu = st.number_input('ARPU Segment (ARPU_SEGMENT)', min_value=0.0)
data_vol = st.number_input('Volume données (DATA_VOLUME)', min_value=0.0)
on_net = st.number_input('Appels on-net (ON_NET)', min_value=0.0)
orange = st.number_input('Appels Orange (ORANGE)', min_value=0.0)
tigo = st.number_input('Appels Tigo (TIGO)', min_value=0.0)
freq_top = st.number_input('Fréquence top pack (FREQ_TOP_PACK)', min_value=0.0)
regularity= st.number_input('Regularite (REGULARITY)', min_value=0.0)

# Encodage des données
tenure_encoded = le_tenure.transform([tenure])[0]

# Création du DataFrame d'entrée
input_data = pd.DataFrame({
    'TENURE': [tenure_encoded],
    'MONTANT': [montant],
    'FREQUENCE_RECH': [freq_rech],
    'ARPU_SEGMENT': [arpu],
    'DATA_VOLUME': [data_vol],
    'ON_NET': [on_net],
    'ORANGE': [orange],
    'TIGO': [tigo],
    'MRG': [0],  # Valeur fixe comme dans le dataset,
    'REGULARITY': [regularity],
    'FREQ_TOP_PACK': [freq_top],

})

# Bouton de prédiction
if st.button('Prédire le Churn'):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Résultat")
    if prediction[0] == 1:
        st.error(f"Risque de désabonnement élevé ({proba * 100:.1f}%) ❗")
    else:
        st.success(f"Client fidèle ({proba * 100:.1f}%) ✅")
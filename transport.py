import requests
import pandas as pd

# URL de l'API SNCF
url = "https://api.sncf.com/v1/coverage/sncf/journeys"
headers = {
    "Authorization": "VotreCléAPI"  # Remplacez par votre clé API
}

# Paramètres pour la requête
params = {
    "from": "stop_area:SNCF:87686006",  # Gare d'origine (ex : Paris Gare de Lyon)
    "to": "stop_area:SNCF:87722025",    # Gare de destination (ex : Lyon Part-Dieu)
    "datetime": "20250117T080000",      # Date et heure
    "max_journeys": 5                   # Nombre de trajets à récupérer
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    # Convertir les données en DataFrame
    journeys = pd.json_normalize(data['journeys'])
    print(journeys.head())
else:
    print("Erreur API :", response.status_code)

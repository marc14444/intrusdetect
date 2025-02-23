from pymongo import MongoClient, errors
from dotenv import load_dotenv
import os

load_dotenv()  # Charger les variables d'environnement depuis le fichier .env

# 🔗 URI de connexion à MongoDB (⚠️ Modifie selon ton setup)
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # Timeout 5 sec
    db = client["siade"]  # 📂 Base de données "siade"
    alerts_collection = db["alerts"]  # 📌 Collection "alerts"
    client.server_info()  # Vérifier la connexion
    print("✅ Connexion réussie à MongoDB")
except errors.ServerSelectionTimeoutError:
    print("❌ ERREUR: Impossible de se connecter à MongoDB. Vérifie ton URI ou connexion réseau.")
    alerts_collection = None  # Si MongoDB est indisponible

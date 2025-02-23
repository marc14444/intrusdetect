from pymongo import MongoClient, errors

# 🔗 URI de connexion à MongoDB (⚠️ Modifie selon ton setup)
MONGO_URI = "mongodb+srv://amanijeanmarc41:siade@siade.vddar.mongodb.net/"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # Timeout 5 sec
    db = client["siade"]  # 📂 Base de données "siade"
    alerts_collection = db["alerts"]  # 📌 Collection "alerts"
    client.server_info()  # Vérifier la connexion
    print("✅ Connexion réussie à MongoDB")
except errors.ServerSelectionTimeoutError:
    print("❌ ERREUR: Impossible de se connecter à MongoDB. Vérifie ton URI ou connexion réseau.")
    alerts_collection = None  # Si MongoDB est indisponible

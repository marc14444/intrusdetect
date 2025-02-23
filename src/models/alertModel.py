from datetime import datetime
from models.database import alerts_collection  # 📂 Importation de la connexion MongoDB

def save_alert(object_type, confidence, bbox, speed, is_running, frame, video_path):
    """📌 Enregistre une alerte dans MongoDB"""
    if alerts_collection is None:
        print("⚠️ Alerte non enregistrée (MongoDB non connecté)")
        return None
    
    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "object_type": object_type,
        "confidence": round(confidence, 2),  
        "bbox": bbox,
        "speed": round(speed, 2),  
        "is_running": bool(is_running),  
        "threat_level": get_threat_level(object_type, speed, is_running),
        "frame": frame,
        "video_path": video_path  # 🔥 Ajout du chemin de la vidéo analysée
    }

    try:
        alerts_collection.insert_one(alert)  # Enregistrement dans MongoDB
        print(f"🚨 ALERTE ENREGISTRÉE: {alert}")
        return alert
    except Exception as e:
        print(f"❌ ERREUR MongoDB: {e}")
        return None

def get_threat_level(object_type, speed, is_running):
    """⚠️ Détermine le niveau de menace"""
    if object_type in ["Couteau", "Pistolet"]:
        return "ÉLEVÉ"
    if object_type in ["Voiture", "Moto", "Camion"] and speed > 5.0:
        return "MOYEN"
    if is_running:
        return "FAIBLE"
    return "AUCUNE"

from flask import Blueprint, request, jsonify
from controllers.detect_intruder_video import IntruderDetector
import os
import uuid
import logging
from models.database import alerts_collection  # ✅ Import unique

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

detection_api = Blueprint("detection_api", __name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@detection_api.route("/detect_video", methods=["POST"])
def detect_video():
    """
    📹 API pour détecter les intrusions dans une vidéo.
    - Enregistre la vidéo reçue, exécute la détection et retourne les résultats.
    """
    try:
        if "video" not in request.files:
            return jsonify({"status": "error", "message": "Aucune vidéo reçue"}), 400

        file = request.files["video"]
        if file.filename == '' or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return jsonify({"status": "error", "message": "Format de fichier non supporté"}), 400

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        logger.info(f"📂 Vidéo reçue et enregistrée: {filepath}")

        # Instanciation du détecteur et analyse de la vidéo
        detector = IntruderDetector()
        result = detector.detect_intruder_in_video(filepath)

        # Vérifier les alertes stockées dans MongoDB
        recent_alerts = list(alerts_collection.find().sort("timestamp", -1).limit(5))
        for alert in recent_alerts:
            alert["_id"] = str(alert["_id"])  # Convertir ObjectId en string pour le JSON

        result["file_path"] = filepath
        result["recent_alerts"] = recent_alerts  # Ajout des alertes récentes dans la réponse
        
        logger.info("✅ Détection terminée et alertes récupérées")
        return jsonify(result)

    except Exception as e:
        logger.exception(f"🚨 Erreur lors du traitement de la vidéo: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@detection_api.route("/alerts", methods=["GET"])
def get_alerts():
    """
    📌 API pour récupérer les alertes stockées dans MongoDB.
    - Permet le filtrage par type d'objet (`object_type`).
    - Gère la pagination avec `limit` et `skip`.
    """
    try:
        object_type = request.args.get("object_type")  # Filtrage optionnel
        limit = int(request.args.get("limit", 20))  # Nombre max d'alertes renvoyées
        skip = int(request.args.get("skip", 0))  # Nombre d'alertes à ignorer

        query = {}  
        if object_type:
            query["object_type"] = object_type  

        alerts = list(alerts_collection.find(query).skip(skip).limit(limit))

        # Conversion de ObjectId en string pour le JSON
        for alert in alerts:
            alert["_id"] = str(alert["_id"])

        logger.info(f"📊 {len(alerts)} alertes récupérées")
        return jsonify({"status": "success", "alerts": alerts}), 200

    except Exception as e:
        logger.error(f"🚨 Erreur lors de la récupération des alertes: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

import cv2
import torch
import numpy as np
import logging
import sys
import os
from ultralytics import YOLO

# Ajouter le chemin src au sys.path pour éviter les erreurs d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Charger le modèle YOLO directement ici (évite les problèmes d'import)
MODEL_PATH = os.path.join("models", "yolov8x.pt")  # Assure-toi que le modèle est bien ici
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = YOLO(MODEL_PATH).to(device)
    logger.info(f"✅ Modèle YOLOv8 chargé avec succès sur {device.upper()} !")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
    sys.exit(1)

# 🚨 Classes dangereuses à détecter (Personnes + Armes + Véhicules dangereux)
DANGEROUS_CLASSES = {
    0: "Personne",
    1: "Vélo", 
    2: "Voiture", 
    3: "Moto", 
    5: "Camion",
    6: "Bus",
    7: "Train",
    8: "Avion",
    49: "Couteau",   # Si modèle personnalisé
    67: "Pistolet",  # Si modèle personnalisé
    70: "Arme",      # Si modèle personnalisé
    71: "Batte"      # Si modèle personnalisé
    
}

def detect_live():
    """🚀 Détection en temps réel avec la webcam"""
    cap = cv2.VideoCapture(0)  # 0 = Webcam par défaut

    if not cap.isOpened():
        logger.error("❌ Impossible d'ouvrir la webcam !")
        return
    
    logger.info("🎥 Détection en temps réel activée... (Appuie sur 'Q' pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ Erreur lors de la capture vidéo")
            break
        
        # 🔍 Détection avec YOLO
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        
        for obj in detections:
            x1, y1, x2, y2 = map(int, obj[:4])
            confidence = float(obj[4])
            class_id = int(obj[5])

            if class_id not in DANGEROUS_CLASSES:
                continue  # Ignorer les objets non pertinents

            object_type = DANGEROUS_CLASSES[class_id]
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # 🟩 Vert pour personne, 🟥 Rouge pour danger
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{object_type} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Détection en Temps Réel - YOLOv8", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("🛑 Détection en temps réel arrêtée.")

if __name__ == "__main__":
    detect_live()

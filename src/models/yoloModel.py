""" from ultralytics import YOLO
import os
import torch

# Définir le chemin du modèle
MODEL_PATH = os.path.join("models", "yolov8n.pt")

# Vérifier si CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le modèle YOLOv8 avec gestion de type
try:
    model = YOLO(MODEL_PATH).to(device)  # NE PAS utiliser `.half()`
    print(f"✅ Modèle YOLOv8 chargé avec succès sur {device.upper()} (dtype=float32) !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
 """

from ultralytics import YOLO
import os
import torch

# Définir le chemin du modèle
MODEL_DIR = "models"
MODEL_NAME = "yolov8x.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Vérifier si CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Télécharger le modèle s'il n'existe pas encore
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        torch.hub.download_url_to_file(
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8x.pt", 
            MODEL_PATH
        )
        print(f"✅ {MODEL_NAME} téléchargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du modèle : {e}")

# Charger le modèle YOLOv8 avec gestion de type
try:
    model = YOLO(MODEL_PATH).to(device)  # NE PAS utiliser `.half()`
    print(f"✅ Modèle {MODEL_NAME} chargé avec succès sur {device.upper()} (dtype=float32) !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
  
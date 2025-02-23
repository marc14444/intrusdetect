import cv2
import numpy as np
import os
from models.yoloModel import model
from datetime import datetime

def detect_intruder(image_path, confidence_threshold=0.5, save_annotated=True):
    try:
        # Charger l'image avec OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": f"Impossible de charger l'image: {image_path}"}
            
        # Créer une copie pour l'annotation
        annotated_image = image.copy()
        
        # Récupérer les dimensions de l'image
        height, width = image.shape[:2]
        
        # Effectuer la détection avec YOLOv8
        results = model(image)

        # Récupérer les objets détectés
        detections = results[0].boxes.data.cpu().numpy()

        # Filtrer pour détecter les humains (ID 0 dans COCO dataset) avec confiance suffisante
        persons_detected = [obj for obj in detections if int(obj[5]) == 0 and obj[4] >= confidence_threshold]  

        # Déterminer s'il y a une intrusion
        person_count = len(persons_detected)
        intrusion_detected = person_count > 0
        
        # Préparer des informations détaillées sur chaque détection
        detection_details = []
        
        # Dessiner les rectangles pour chaque personne détectée
        for obj in persons_detected:
            x1, y1, x2, y2, confidence, class_id = obj.tolist() if hasattr(obj, 'tolist') else list(obj)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculer le pourcentage de la surface de l'image occupée par la détection
            box_area = (x2 - x1) * (y2 - y1)
            image_area = width * height
            area_percentage = (box_area / image_area) * 100
            
            # Dessiner le rectangle (rouge)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Ajouter le texte avec le niveau de confiance
            confidence_text = f"Personne: {confidence:.2f}"
            cv2.putText(annotated_image, confidence_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            detection_details.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": "personne",
                "area_percentage": float(area_percentage)
            })

        # Sauvegarder l'image annotée si demandé
        annotated_image_path = None
        if save_annotated and intrusion_detected:
            # Générer un nom de fichier basé sur l'horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            
            # Créer le dossier de sortie s'il n'existe pas
            output_dir = os.path.join(os.path.dirname(image_path), "detections")
            os.makedirs(output_dir, exist_ok=True)
            
            # Chemin complet pour l'image annotée
            annotated_image_path = os.path.join(output_dir, f"{base_name}_detection_{timestamp}{ext}")
            
            # Sauvegarder l'image
            cv2.imwrite(annotated_image_path, annotated_image)

        # Ajouter un horodatage
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "timestamp": timestamp,
            "message": f"{person_count} intrus détecté{'s' if person_count > 1 else ''} !" if intrusion_detected else "Aucune intrusion détectée",
            "intrusion_detected": intrusion_detected,
            "person_count": person_count,
            "confidence_threshold": confidence_threshold,
            "detections": detection_details,
            "annotated_image_path": annotated_image_path
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
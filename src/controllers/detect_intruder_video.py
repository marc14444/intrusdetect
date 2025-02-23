import cv2
import os
import numpy as np
import json
from datetime import datetime
import logging
from models.yoloModel import model
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from models.alertModel import save_alert
from models.database import alerts_collection  # Correction ajoutée

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# 🚨 CLASSES D'OBJETS DANGEREUX (Personnes + Armes + Véhicules)
DANGEROUS_CLASSES = {
    0: "Personne",
    49: "Couteau",
    67: "Pistolet",
}

@dataclass
class Detection:
    frame: int
    time: float
    bbox: List[int]
    confidence: float
    is_running: bool
    speed: float
    object_type: str

@dataclass
class DetectionReport:
    status: str
    intrusion_detected: bool
    detections: List[Dict[str, Any]]
    video_path: Optional[str]
    processing_time: float
    total_persons_detected: int  

class IntruderDetector:
    def __init__(self):
        self.previous_positions = {}  # Stocker les positions des objets des frames précédentes
        self.running_threshold = 2.5  # Seuil de vitesse pour détecter la course

    def _prepare_output_paths(self, video_path: str) -> Tuple[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(UPLOADS_DIR, "detections", f"{base_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{base_name}_detection.mp4")
        return output_dir, output_video_path

    def detect_intruder_in_video(self, video_path: str) -> Dict[str, Any]:
        start_time = datetime.now()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Impossible de charger la vidéo: {video_path}")
            return {"status": "error", "message": "Fichier vidéo inaccessible"}

        fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)  # Éviter la division par zéro
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_dir, output_video_path = self._prepare_output_paths(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        total_persons_detected = 0
        detection_details = []
        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analyse de la vidéo", unit="frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

            current_positions = {}  

            for obj in detections:
                x1, y1, x2, y2 = map(int, obj[:4])
                confidence = float(obj[4])
                class_id = int(obj[5])

                if class_id not in DANGEROUS_CLASSES:
                    continue  

                object_type = DANGEROUS_CLASSES[class_id]

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                object_key = f"{class_id}_{center_x}_{center_y}"

                speed = 0.0
                is_running = False

                if object_key in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[object_key]
                    distance = np.linalg.norm([center_x - prev_x, center_y - prev_y])
                    speed = (distance / fps) * 30  # Normalisation de la vitesse

                    if speed > self.running_threshold:
                        is_running = True

                current_positions[object_key] = (center_x, center_y)

                if class_id == 0:
                    total_persons_detected += 1

                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                if is_running:
                    color = (0, 165, 255)  # Orange si la personne court

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{object_type} {confidence:.2f} | Speed: {speed:.2f} m/s", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detection = Detection(
                    frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    time=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / fps,
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    is_running=is_running,
                    speed=speed,
                    object_type=object_type
                )
                detection_details.append(asdict(detection))

                alert = save_alert(object_type, confidence, [x1, y1, x2, y2], speed, is_running,
                                   frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES)), video_path=output_video_path)
                logger.info(f"🔴 ALERTE SAUVEGARDÉE: {alert}")

            self.previous_positions = current_positions
            out.write(frame)
            progress_bar.update(1)

        cap.release()
        out.release()
        progress_bar.close()

        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "status": "success",
            "video_path": output_video_path,
            "total_persons_detected": total_persons_detected,
            "detections": detection_details,
            "processing_time": processing_time
        }

        return json.loads(json.dumps(result, default=lambda o: bool(o) if isinstance(o, np.bool_) else o))

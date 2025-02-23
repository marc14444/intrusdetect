import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math

@dataclass
class MotionData:
    is_moving: bool
    is_running: bool
    speed: float
    direction: Tuple[float, float]
    contour_area: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None

class MovementType(Enum):
    STATIONARY = 0
    WALKING = 1
    RUNNING = 2
    ERRATIC = 3

class MotionDetector:
    def __init__(self, 
                 threshold: int = 25,
                 min_area: int = 800,
                 history_size: int = 10,
                 running_threshold: float = 20.0,
                 background_subtractor: str = 'MOG2'):
        self.threshold = threshold
        self.min_area = min_area
        self.history_size = history_size
        self.running_threshold = running_threshold

        self.position_history = []
        self.speed_history = []
        self.prev_frame = None  # Correction ici

        if background_subtractor == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        elif background_subtractor == 'KNN':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=400, detectShadows=False)
        else:
            self.bg_subtractor = None

        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_points = None

    def _apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Applique la soustraction de fond pour isoler les objets en mouvement."""
        if self.bg_subtractor:
            fg_mask = self.bg_subtractor.apply(frame)
            _, thresh = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)
            return cv2.dilate(thresh, None, iterations=2)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if self.prev_frame is None:
                self.prev_frame = gray
                return np.zeros_like(gray)

            frame_diff = cv2.absdiff(self.prev_frame, gray)
            self.prev_frame = gray

            _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
            return cv2.dilate(thresh, None, iterations=2)

    def _calculate_optical_flow(self, frame: np.ndarray) -> Tuple[List[Tuple[float, float]], List[bool]]:
        """Calcule le flot optique pour suivre les mouvements."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return [], []

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.prev_points is None:
                self.prev_gray = gray
                return [], []

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        if new_points is None or status is None:
            self.prev_gray = gray
            return [], []

        valid_idx = status.ravel() == 1
        good_new = new_points[valid_idx]
        good_old = self.prev_points[valid_idx]

        flows = [(a - c, b - d) for (a, b), (c, d) in zip(good_new.reshape(-1, 2), good_old.reshape(-1, 2))]
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)

        return flows, valid_idx.tolist()

    def analyze_motion(self, frame: np.ndarray) -> MotionData:
        """Détecte et analyse le mouvement dans une frame."""
        fg_mask = self._apply_background_subtraction(frame)
        flows, valid_points = self._calculate_optical_flow(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        bounding_box = None
        current_position = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area and area > max_area:
                max_area = area
                x, y, w, h = cv2.boundingRect(contour)
                bounding_box = (x, y, w, h)
                current_position = (x + w // 2, y + h // 2)

        avg_speed = sum(math.hypot(dx, dy) for dx, dy in flows) / len(flows) if flows else 0
        is_running = avg_speed > self.running_threshold

        return MotionData(
            is_moving=bool(max_area > 0),
            is_running=is_running,
            speed=avg_speed,
            direction=(0, 0),  # Ajout d'une direction par défaut
            contour_area=max_area,
            bounding_box=bounding_box
        )

    def _visualize_motion(self, frame: np.ndarray, motion_data: MotionData) -> np.ndarray:
        """Affiche la détection de mouvement."""
        result_frame = frame.copy()

        if motion_data.is_moving and motion_data.bounding_box:
            x, y, w, h = motion_data.bounding_box
            color = (0, 0, 255) if motion_data.is_running else (0, 255, 0)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_frame, f"Speed: {motion_data.speed:.1f} px/frame", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result_frame

```python
import cv2
import numpy as np
import math
import json
import yaml
import os
import time
from typing import List, Tuple, Dict, Any, Optional
import mediapipe as mp

# MediaPipe setup (convenience)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Constants ---
VISUALIZATION_CIRCLE_RADIUS = 5
VISUALIZATION_LINE_THICKNESS = 2
VISUALIZATION_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
VISUALIZATION_TEXT_SCALE = 0.8
VISUALIZATION_TEXT_THICKNESS = 2
VISUALIZATION_TEXT_COLOR = (255, 255, 255)
VISUALIZATION_LANDMARK_COLOR = (0, 255, 0)
VISUALIZATION_CONNECTION_COLOR = (0, 0, 255)
VISUALIZATION_BBOX_COLOR = (255, 0, 0)
DEFAULT_CONFIG_PATH = 'config.yaml'

# --- Data Visualization ---

def draw_landmarks(
    image: np.ndarray,
    hand_landmarks: Any, # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    connections: Optional[List[Tuple[int, int]]] = mp_hands.HAND_CONNECTIONS,
    landmark_drawing_spec: mp_drawing.DrawingSpec = mp_drawing_styles.get_default_hand_landmarks_style(),
    connection_drawing_spec: mp_drawing.DrawingSpec = mp_drawing_styles.get_default_hand_connections_style()
) -> np.ndarray:
    annotated_image = image.copy()
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            connections,
            landmark_drawing_spec,
            connection_drawing_spec
        )
    return annotated_image

def draw_bounding_box(
    image: np.ndarray,
    hand_landmarks: Any # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    if not hand_landmarks or not hand_landmarks.landmark:
        return image, None

    image_height, image_width, _ = image.shape
    try:
        x_coords = [landmark.x * image_width for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * image_height for landmark in hand_landmarks.landmark]
    except AttributeError:
         # Handle cases where landmark structure might be different or empty
        return image, None

    if not x_coords or not y_coords:
        return image, None

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width - 1, x_max + padding)
    y_max = min(image_height - 1, y_max + padding)

    bbox = (x_min, y_min, x_max - x_min, y_max - y_min) # x, y, w, h

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), VISUALIZATION_BBOX_COLOR, VISUALIZATION_LINE_THICKNESS)
    return image, bbox


def draw_info_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font: int = VISUALIZATION_TEXT_FONT,
    scale: float = VISUALIZATION_TEXT_SCALE,
    color: Tuple[int, int, int] = VISUALIZATION_TEXT_COLOR,
    thickness: int = VISUALIZATION_TEXT_THICKNESS,
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    bg_alpha: float = 0.6
) -> np.ndarray:
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position
    text_y = y + text_h // 2 # Adjust position to be more intuitive (top-left corner)

    if bg_color:
        rect_x1 = x - 2
        rect_y1 = text_y - text_h - baseline - 2
        rect_x2 = x + text_w + 2
        rect_y2 = text_y + baseline + 2

        # Ensure rectangle coordinates are within image bounds
        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)
        rect_x2 = min(image.shape[1] - 1, rect_x2)
        rect_y2 = min(image.shape[0] - 1, rect_y2)

        if rect_x1 < rect_x2 and rect_y1 < rect_y2: # Check if rectangle is valid
            sub_img = image[rect_y1:rect_y2, rect_x1:rect_x2]
            bg_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            bg_rect[:] = bg_color
            res = cv2.addWeighted(sub_img, 1 - bg_alpha, bg_rect, bg_alpha, 0)
            image[rect_y1:rect_y2, rect_x1:rect_x2] = res

    cv2.putText(image, text, (x, text_y), font, scale, color, thickness, cv2.LINE_AA)
    return image


# --- Metrics Calculation ---

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.dist(point1, point2)

def calculate_landmark_distance(lm1: Any, lm2: Any) -> float:
     # mediapipe.framework.formats.landmark_pb2.NormalizedLandmark or similar structure with x, y, z
    try:
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    except AttributeError:
        return float('inf') # Indicate error or invalid input

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    # p1: end point, p2: vertex (joint), p3: start point
    # Create vectors
    v1 = p1 - p2 # Vector from vertex to end point
    v2 = p3 - p2 # Vector from vertex to start point

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Prevent division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Calculate cosine of the angle
    cosine_angle = dot_product / (norm_v1 * norm_v2)

    # Clip the value to [-1, 1] due to potential floating point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def get_landmark_coordinates(
    hand_landmarks: Any, # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    landmark_id: int,
    image_shape: Tuple[int, int, int] # height, width, channels
) -> Optional[Tuple[int, int]]:
    if not hand_landmarks or not hand_landmarks.landmark or landmark_id >= len(hand_landmarks.landmark):
        return None
    landmark = hand_landmarks.landmark[landmark_id]
    image_height, image_width, _ = image_shape
    # Check if landmark has valid coordinates
    if landmark.x is None or landmark.y is None:
        return None
    coord_x = int(landmark.x * image_width)
    coord_y = int(landmark.y * image_height)
    return coord_x, coord_y

def get_all_landmark_coordinates(
    hand_landmarks: Any, # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    image_shape: Tuple[int, int, int] # height, width, channels
) -> Optional[List[Tuple[int, int]]]:
    if not hand_landmarks or not hand_landmarks.landmark:
        return None
    image_height, image_width, _ = image_shape
    coords = []
    for landmark in hand_landmarks.landmark:
         # Check if landmark has valid coordinates
        if landmark.x is None or landmark.y is None:
            coords.append(None) # Keep list length consistent, indicate missing data
        else:
            coord_x = int(landmark.x * image_width)
            coord_y = int(landmark.y * image_height)
            coords.append((coord_x, coord_y))
    # Return None if no valid coordinates were found at all
    return coords if any(c is not None for c in coords) else None

def normalize_landmarks_relative_to_wrist(hand_landmarks: Any) -> Optional[List[Tuple[float, float, float]]]:
    if not hand_landmarks or not hand_landmarks.landmark:
        return None

    try:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        if wrist.x is None or wrist.y is None or wrist.z is None:
             return None # Wrist data is invalid

        normalized_landmarks = []
        for landmark in hand_landmarks.landmark:
            if landmark.x is None or landmark.y is None or landmark.z is None:
                normalized_landmarks.append((float('nan'), float('nan'), float('nan'))) # Indicate missing data
            else:
                normalized_landmarks.append(
                    (landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z)
                )
        return normalized_landmarks
    except (AttributeError, IndexError):
        return None # Handle cases where landmarks or wrist might be missing


# --- File Operations ---

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        # print(f"Warning: File not found at {filepath}") # Suppress output for production
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        # print(f"Error decoding JSON from {filepath}: {e}") # Suppress output
        return None
    except Exception as e:
        # print(f"Error loading JSON file {filepath}: {e}") # Suppress output
        return None

def save_json(data: Dict[str, Any], filepath: str) -> bool:
    try:
        dir_path = os.path.dirname(filepath)
        if dir_path: # Ensure directory exists only if path includes one
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        # print(f"Error saving JSON file {filepath}: {e}") # Suppress output
        return False

def load_yaml(filepath: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        # print(f"Warning: File not found at {filepath}") # Suppress output
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Check if loading resulted in None (e.g., empty file) which is valid YAML but maybe not intended config
        return data if data is not None else {}
    except yaml.YAMLError as e:
        # print(f"Error decoding YAML from {filepath}: {e}") # Suppress output
        return None
    except Exception as e:
        # print(f"Error loading YAML file {filepath}: {e}") # Suppress output
        return None

def save_yaml(data: Dict[str, Any], filepath: str) -> bool:
    try:
        dir_path = os.path.dirname(filepath)
        if dir_path: # Ensure directory exists only if path includes one
             os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        # print(f"Error saving YAML file {filepath}: {e}") # Suppress output
        return False

# --- Configuration Management ---

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config = {}
    if config_path.endswith(('.yaml', '.yml')):
        config = load_yaml(config_path)
    elif config_path.endswith('.json'):
        config = load_json(config_path)
    else:
        # Attempt loading as YAML first, then JSON if file exists but has unknown extension
        if os.path.exists(config_path):
            config = load_yaml(config_path)
            if config is None:
                config = load_json(config_path)
        # else: # Suppress output
            # print(f"Warning: Config file {config_path} not found.")

    # Return empty dict if loading failed or file was empty
    return config if config is not None else {}

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            if isinstance(value, dict):
                value = value[k]
            elif isinstance(value, list) and k.isdigit(): # Allow indexing into lists
                 idx = int(k)
                 if 0 <= idx < len(value):
                     value = value[idx]
                 else:
                     # print(f"Warning: Index '{k}' out of bounds for key path '{key}'. Returning default.") # Suppress
                     return default
            else:
                # print(f"Warning: Key path '{key}' segment '{k}' not found or not traversable. Returning default.") # Suppress
                return default
        return value
    except (KeyError, TypeError, IndexError):
        # print(f"Warning: Key '{key}' not found or invalid path in config. Returning default.") # Suppress
        return default
    except Exception as e:
        # print(f"Error accessing config key '{key}': {e}. Returning default.") # Suppress
        return default


# --- Other Utilities ---

class FpsCalculator:
    def __init__(self, smoothing_factor
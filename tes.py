import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import math
import time
from pygame.locals import *
from ursina import *
from ursina import color as ursina_color

WEBCAM_INDEX = 0
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SIM_AREA_WIDTH_RATIO = 0.6
VIDEO_AREA_WIDTH_RATIO = 1.0 - SIM_AREA_WIDTH_RATIO

BG_COLOR = (30, 30, 30)
SIM_BG_COLOR = (40, 40, 50)
VIDEO_BG_COLOR = (40, 40, 40)
TEXT_COLOR = (230, 230, 230)
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 0, 255)
GESTURE_TEXT_COLOR = (255, 255, 0)

DRONE_COLOR = (0, 180, 255)
DRONE_SIZE = 40
DRONE_SPEED = 5
DRONE_ROTATION_SPEED = 3
DRONE_VERTICAL_SPEED = 3

ENABLE_3D = False
ENABLE_2D = not ENABLE_3D
GRID_SIZE = 20
GRID_COUNT = 10

ENABLE_TRAIL = True
TRAIL_LENGTH = 50
TRAIL_COLOR = (80, 130, 200, 150)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class GestureRecognizer:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1)
        self.results = None
        self.image_height = 0
        self.image_width = 0
        self.smoothed_landmarks = None
        self.alpha = 0.7
        self.gesture_history = []
        self.history_size = 5
        self.current_gesture = "NO_HAND"
        
    def process_frame(self, image_rgb):
        self.image_height, self.image_width = image_rgb.shape[:2]
        self.results = self.hands.process(image_rgb)
        
        if self.results and self.results.multi_hand_landmarks:
            new_landmarks = self.results.multi_hand_landmarks[0].landmark
            
            if self.smoothed_landmarks is None:
                self.smoothed_landmarks = [
                    {'x': l.x, 'y': l.y, 'z': l.z} for l in new_landmarks
                ]
            else:
                for i, landmark in enumerate(new_landmarks):
                    self.smoothed_landmarks[i]['x'] = self.alpha * self.smoothed_landmarks[i]['x'] + (1 - self.alpha) * landmark.x
                    self.smoothed_landmarks[i]['y'] = self.alpha * self.smoothed_landmarks[i]['y'] + (1 - self.alpha) * landmark.y
                    self.smoothed_landmarks[i]['z'] = self.alpha * self.smoothed_landmarks[i]['z'] + (1 - self.alpha) * landmark.z
        else:
            self.smoothed_landmarks = None

    def get_landmarks(self):
        if self.smoothed_landmarks:
            return self.smoothed_landmarks
        elif self.results and self.results.multi_hand_landmarks:
            return self.results.multi_hand_landmarks[0].landmark
        return None

    def _vector_angle(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)
    
    def _finger_is_curved(self, finger_landmarks, threshold=0.4):
        tip = np.array([finger_landmarks[0]['x'], finger_landmarks[0]['y']])
        mid = np.array([finger_landmarks[1]['x'], finger_landmarks[1]['y']])
        base = np.array([finger_landmarks[2]['x'], finger_landmarks[2]['y']])
        
        v1 = mid - base
        v2 = tip - mid
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return False
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        angle_deg = np.degrees(angle)
        
        return angle_deg > threshold * 180

    def recognize(self, landmarks):
        if not landmarks:
            return "NO_HAND", None, 0.0
            
        wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST]['x'], landmarks[mp_hands.HandLandmark.WRIST]['y']])
        thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP]['x'], landmarks[mp_hands.HandLandmark.THUMB_TIP]['y']])
        thumb_ip = np.array([landmarks[mp_hands.HandLandmark.THUMB_IP]['x'], landmarks[mp_hands.HandLandmark.THUMB_IP]['y']])
        thumb_mcp = np.array([landmarks[mp_hands.HandLandmark.THUMB_MCP]['x'], landmarks[mp_hands.HandLandmark.THUMB_MCP]['y']])
        
        index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'], landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y']])
        index_pip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]['x'], landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]['y']])
        index_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]['x'], landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]['y']])
        
        middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]['x'], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]['y']])
        middle_pip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]['x'], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]['y']])
        middle_mcp = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]['x'], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]['y']])
        
        ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]['x'], landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]['y']])
        ring_pip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]['x'], landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]['y']])
        ring_mcp = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]['x'], landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]['y']])
        
        pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP]['x'], landmarks[mp_hands.HandLandmark.PINKY_TIP]['y']])
        pinky_pip = np.array([landmarks[mp_hands.HandLandmark.PINKY_PIP]['x'], landmarks[mp_hands.HandLandmark.PINKY_PIP]['y']])
        pinky_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP]['x'], landmarks[mp_hands.HandLandmark.PINKY_MCP]['y']])

        palm_center = np.mean([index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)
        
        palm_size = np.linalg.norm(wrist - middle_mcp)
        if palm_size < 1e-6: palm_size = 1e-6

        z_value = landmarks[mp_hands.HandLandmark.WRIST]['z']
        hand_depth = z_value

        palm_normal = np.cross(
            np.append(middle_mcp - index_mcp, 0), 
            np.append(ring_mcp - index_mcp, 0)
        )
        palm_facing_camera = palm_normal[2] < 0

        fingers = [
            [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], landmarks[mp_hands.HandLandmark.RING_FINGER_PIP], landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.PINKY_TIP], landmarks[mp_hands.HandLandmark.PINKY_PIP], landmarks[mp_hands.HandLandmark.PINKY_MCP]]
        ]
        
        finger_curled = [self._finger_is_curved(finger) for finger in fingers]
        
        thumb_curled = np.linalg.norm(thumb_tip - index_mcp) < np.linalg.norm(wrist - index_mcp) * 0.4
        
        fingers_curled = [thumb_curled] + finger_curled
        fingers_extended = [not curled for curled in fingers_curled]

        pinch_distance = np.linalg.norm(thumb_tip - index_tip) / palm_size
        is_pinching = pinch_distance < 0.15

        pointing_vector = None
        pointing_intensity = 0.0
        
        if fingers_extended[1]:
            pointing_vector = index_tip - index_pip
            pointing_magnitude = np.linalg.norm(pointing_vector)
            
            finger_straightness = 1.0 - min(1.0, self._vector_angle(index_tip - index_pip, index_pip - index_mcp) / 40.0)
            pointing_intensity = pointing_magnitude * finger_straightness
            
            if pointing_magnitude > 1e-6:
                pointing_vector /= pointing_magnitude

        if all(fingers_curled):
            return "FIST", None, 0.0
            
        elif all(fingers_extended):
            return "OPEN_PALM", None, 0.0
            
        elif fingers_extended[0] and not any(fingers_extended[1:]) and not palm_facing_camera:
            return "THUMBS_UP", None, 0.0
            
        elif fingers_extended[0] and not any(fingers_extended[1:]) and palm_facing_camera:
            return "THUMBS_DOWN", None, 0.0
            
        elif is_pinching:
            pinch_center = (thumb_tip + index_tip) / 2
            pinch_vector = pinch_center - palm_center
            pinch_magnitude = np.linalg.norm(pinch_vector)
            if pinch_magnitude > 1e-6:
                pinch_vector /= pinch_magnitude
            return "PINCH", pinch_vector, pinch_distance
            
        elif fingers_extended[1] and not any([fingers_extended[0]] + fingers_extended[2:]):
            if pointing_vector is not None:
                angle = math.atan2(-pointing_vector[1], pointing_vector[0]) * 180 / math.pi
                if -45 < angle <= 45:
                    return "POINTING_RIGHT", pointing_vector, pointing_intensity
                elif 45 < angle <= 135:
                    return "POINTING_UP", pointing_vector, pointing_intensity
                elif 135 < angle or angle <= -135:
                    return "POINTING_LEFT", pointing_vector, pointing_intensity
                elif -135 < angle <= -45:
                    return "POINTING_DOWN", pointing_vector, pointing_intensity
            return "POINTING", pointing_vector, pointing_intensity
            
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not any(fingers_extended[3:]):
            return "PEACE", None, 0.0
            
        elif fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[0] and not fingers_extended[4]:
            return "THREE", None, 0.0
            
        elif fingers_extended[1] and fingers_extended[4] and not any(fingers_extended[0:1] + fingers_extended[2:4]):
            return "ROCK_ON", None, 0.0
        
        elif fingers_extended[0] and fingers_extended[4] and not any(fingers_extended[1:4]):
            return "CALL_ME", None, 0.0
            
        elif is_pinching and all(fingers_extended[2:]):
            return "OK_SIGN", None, 0.0
            
        return "UNKNOWN", None, 0.0

    def get_stable_gesture(self, current_gesture, vector, intensity):
        self.gesture_history.append((current_gesture, vector, intensity))
        
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
            
        gesture_counts = {}
        for g, _, _ in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
        if gesture_counts:
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            if most_common[1] >= self.history_size / 2:
                self.current_gesture = most_common[0]
                for g, v, i in reversed(self.gesture_history):
                    if g == most_common[0]:
                        return g, v, i
        
        self.current_gesture = current_gesture
        return current_gesture, vector, intensity

    def draw_landmarks(self, image):
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            landmarks = self.get_landmarks()
            if landmarks:
                if self.current_gesture == "FIST":
                    cv2.circle(image, 
                            (int(landmarks[mp_hands.HandLandmark.WRIST]['x'] * image.shape[1]),
                             int(landmarks[mp_hands.HandLandmark.WRIST]['y'] * image.shape[0])),
                            15, (0, 0, 255), -1)
                
                elif self.current_gesture == "OPEN_PALM":
                    for tip_id in [mp_hands.HandLandmark.THUMB_TIP, 
                                  mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                  mp_hands.HandLandmark.RING_FINGER_TIP,
                                  mp_hands.HandLandmark.PINKY_TIP]:
                        cv2.circle(image, 
                                (int(landmarks[tip_id]['x'] * image.shape[1]),
                                 int(landmarks[tip_id]['y'] * image.shape[0])),
                                8, (0, 255, 255), -1)
                
                elif "POINTING" in self.current_gesture:
                    index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y'] * image.shape[0]))
                    
                    dx, dy = 0, 0
                    if self.current_gesture == "POINTING_UP": dy = -50
                    elif self.current_gesture == "POINTING_DOWN": dy = 50
                    elif self.current_gesture == "POINTING_LEFT": dx = -50
                    elif self.current_gesture == "POINTING_RIGHT": dx = 50
                    
                    cv2.arrowedLine(image, index_tip, 
                                   (index_tip[0] + dx, index_tip[1] + dy),
                                   (255, 0, 0), 3)
                
                elif self.current_gesture == "PINCH":
                    thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.THUMB_TIP]['y'] * image.shape[0]))
                    index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y'] * image.shape[0]))
                    
                    pinch_center = ((thumb_tip[0] + index_tip[0])//2, (thumb_tip[1] + index_tip[1])//2)
                    cv2.circle(image, pinch_center, 10, (255, 0, 255), -1)
                
                palm_x = int(np.mean([
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]['x'],
                    landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]['x'],
                    landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]['x'],
                    landmarks[mp_hands.HandLandmark.PINKY_MCP]['x']
                ]) * image.shape[1])
                
                palm_y = int(np.mean([
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]['y'],
                    landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]['y'],
                    landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]['y'],
                    landmarks[mp_hands.HandLandmark.PINKY_MCP]['y']
                ]) * image.shape[0])
                
                cv2.circle(image, (palm_x, palm_y), 10, (0, 255, 0), -1)
                
            cv2.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.putText(image, f"Gesture: {self.current_gesture}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
        return image

    def close(self):
        self.hands.close()

class DroneController:
    def __init__(self, sim_area_rect):
        self.sim_rect = sim_area_rect
        
        self.drone_x = self.sim_rect.centerx
        self.drone_y = self.sim_rect.centery
        self.drone_rect = pygame.Rect(0, 0, DRONE_SIZE, DRONE_SIZE)
        self.drone_rect.center = (self.drone_x, self.drone_y)
        
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.target_rotation = [0.0, 0.0, 0.0]
        
        self.speed = DRONE_SPEED
        self.vertical_speed = DRONE_VERTICAL_SPEED
        self.rotation_speed = DRONE_ROTATION_SPEED
        self.target_x = self.drone_x
        self.target_y = self.drone_y
        self.target_z = 0.0
        
        self.command = "STOP"
        self.prev_command = "STOP"
        self.mode = "normal"
        
        self.trail = []
        self.trail_max_length = TRAIL_LENGTH
        
        self.ursina_drone = None
        self.drone_model = None
        self.trail_entities = []
        
    def update_command(self, gesture, vector=None, intensity=0.0):
        self.prev_command = self.command
        
        if gesture == "PEACE":
            self.mode = "normal"
            self.command = "MODE_NORMAL"
            return
        elif gesture == "THREE":
            self.mode = "rotation"
            self.command = "MODE_ROTATION"
            return
        elif gesture == "ROCK_ON":
            self.mode = "height"
            self.command = "MODE_HEIGHT"
            return
        elif gesture == "THUMBS_UP":
            self.command = "SPEED_UP"
            self.speed = min(10, self.speed + 1)
            self.vertical_speed = min(6, self.vertical_speed + 0.5)
            return
        elif gesture == "THUMBS_DOWN":
            self.command = "SPEED_DOWN"
            self.speed = max(1, self.speed - 1)
            self.vertical_speed = max(1, self.vertical_speed - 0.5)
            return
        
        if self.mode == "normal":
            if gesture == "FIST":
                self.command = "STOP"
            elif gesture == "OPEN_PALM":
                self.command = "HOVER"
            elif "POINTING" in gesture:
                self.command = gesture.replace("POINTING_", "MOVE_")
                if vector is not None and gesture != "POINTING":
                    direction = gesture.split("_")[1]
                    intensity_factor = min(5.0, max(1.0, intensity * 3.0))
                    
                    if direction == "UP":
                        self.target_y = max(self.sim_rect.top, self.drone_y - self.speed * intensity_factor)
                    elif direction == "DOWN":
                        self.target_y = min(self.sim_rect.bottom, self.drone_y + self.speed * intensity_factor)
                    elif direction == "LEFT":
                        self.target_x = max(self.sim_rect.left, self.drone_x - self.speed * intensity_factor)
                    elif direction == "RIGHT":
                        self.target_x = min(self.sim_rect.right, self.drone_x + self.speed * intensity_factor)
                    
                    self.target_position = [
                        (self.target_x - self.sim_rect.centerx) / 100.0,
                        0,
                        (self.target_y - self.sim_rect.centery) / 100.0
                    ]
                    
            elif gesture == "PINCH":
                self.command = "FOLLOW"
                if vector is not None:
                    center_x = self.sim_rect.centerx
                    center_y = self.sim_rect.centery
                    
                    norm_x = vector[0]
                    norm_y = vector[1]
                    
                    self.target_x = center_x + norm_x * self.sim_rect.width / 3
                    self.target_y = center_y + norm_y * self.sim_rect.height / 3
                    
                    self.target_position = [
                        (self.target_x - center_x) / 100.0,
                        0,
                        (self.target_y - center_y) / 100.0
                    ]
        
        elif self.mode == "rotation":
            if gesture == "POINTING_LEFT":
                self.command = "ROTATE_LEFT"
                self.target_rotation[1] += self.rotation_speed
            elif gesture == "POINTING_RIGHT":
                self.command = "ROTATE_RIGHT"
                self.target_rotation[1] -= self.rotation_speed
            elif gesture == "POINTING_UP":
                self.command = "ROTATE_UP"
                self.target_rotation[0] -= self.rotation_speed
            elif gesture == "POINTING_DOWN":
                self.command = "ROTATE_DOWN"
                self.target_rotation[0] += self.rotation_speed
            elif gesture == "OPEN_PALM":
                self.command = "RESET_ROTATION"
                self.target_rotation = [0, 0, 0]
        
        elif self.mode == "height":
            if gesture == "POINTING_UP":
                self.command = "ASCEND"
                self.target_z -= self.vertical_speed / 10.0
            elif gesture == "POINTING_DOWN":
                self.command = "DESCEND"
                self.target_z += self.vertical_speed / 10.0
            elif gesture == "OPEN_PALM":
                self.command = "HOVER_HEIGHT"
            elif gesture == "FIST":
                self.command = "RESET_HEIGHT"
                self.target_z = 0.0
    
    def update(self):
        dx = self.target_x - self.drone_x
        dy = self.target_y - self.drone_y
        dz = self.target_z - self.position[1]
        
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 1:
            move_x = dx * min(1, self.speed / distance)
            move_y = dy * min(1, self.speed / distance)
            
            self.drone_x += move_x
            self.drone_y += move_y
            
            self.drone_x = max(self.sim_rect.left, min(self.sim_rect.right, self.drone_x))
            self.drone_y = max(self.sim_rect.top, min(self.sim_rect.bottom, self.drone_y))
            
            self.drone_rect.center = (self.drone_x, self.drone_y)
        
        target_position = [
            (self.drone_x - self.sim_rect.centerx) / 100.0,
            self.target_z,
            (self.drone_y - self.sim_rect.centery) / 100.0
        ]
        
        self.position[0] += (target_position[0] - self.position[0]) * 0.2
        self.position[1] += (target_position[1] - self.position[1]) * 0.2
        self.position[2] += (target_position[2] - self.position[2]) * 0.2
        
        for i in range(3):
            self.rotation[i] += (self.target_rotation[i] - self.rotation[i]) * 0.1
        
        if ENABLE_TRAIL:
            self.trail.append((self.drone_x, self.drone_y))
            if len(self.trail) > self.trail_max_length:
                self.trail.pop(0)
        
        if ENABLE_3D and self.ursina_drone:
            self.ursina_drone.x = self.position[0]
            self.ursina_drone.y = self.position[1]
            self.ursina_drone.z = self.position[2]
            
            self.ursina_drone.rotation_x = self.rotation[0]
            self.ursina_drone.rotation_y = self.rotation[1]
            self.ursina_drone.rotation_z = self.rotation[2]
            
            if ENABLE_TRAIL and hasattr(self, 'trail_entities'):
                for e in self.trail_entities:
                    if e is not None:
                        try:
                            destroy(e)
                        except:
                            pass
                self.trail_entities = []
                
                if len(self.trail) > 1:
                    for i in range(1, len(self.trail)):
                        pos1 = ((self.trail[i-1][0] - self.sim_rect.centerx) / 100.0, 
                                self.position[1], 
                                (self.trail[i-1][1] - self.sim_rect.centery) / 100.0)
                        pos2 = ((self.trail[i][0] - self.sim_rect.centerx) / 100.0, 
                                self.position[1], 
                                (self.trail[i][1] - self.sim_rect.centery) / 100.0)
                        
                        try:
                            # Calculate distance between points
                            dist = ((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 + (pos2[2]-pos1[2])**2)**0.5
                            
                            if dist > 0:  # Prevent issues with zero distance
                                alpha = i / len(self.trail)
                                color = ursina_color.rgba(TRAIL_COLOR[0]/255, TRAIL_COLOR[1]/255, TRAIL_COLOR[2]/255, alpha)
                                
                                # Create cube entity instead of cylinder for simplicity and less model dependency
                                line = Entity(model='cube', 
                                            position=((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2, (pos1[2] + pos2[2])/2), 
                                            scale=(0.05, 0.05, dist),
                                            color=color)
                                
                                # Calculate direction vector
                                dir_x = pos2[0] - pos1[0]
                                dir_y = pos2[1] - pos1[1]
                                dir_z = pos2[2] - pos1[2]
                                
                                # Make entity look in the direction of movement
                                if abs(dir_x) > 0.001 or abs(dir_z) > 0.001:
                                    line.look_at(Vec3(pos2[0], pos2[1], pos2[2]), axis='forward')
                                
                                self.trail_entities.append(line)
                        except Exception as e:
                            print(f"Error creating trail: {e}")
    
def draw(self, surface):
    # Draw the drone in 2D (needed for ENABLE_2D mode or fallback)
    if ENABLE_2D:
        # Draw drone's trail 
        if ENABLE_TRAIL and len(self.trail) > 1:
            if len(self.trail) > 1:
                for i in range(1, len(self.trail)):
                    # Adjust alpha value based on position in trail for fade effect
                    alpha = int(255 * i / len(self.trail))
                    trail_color = (TRAIL_COLOR[0], TRAIL_COLOR[1], TRAIL_COLOR[2], alpha)
                    pygame.draw.line(surface, trail_color, self.trail[i-1], self.trail[i], 3)
        
        # Draw the drone itself
        pygame.draw.rect(surface, DRONE_COLOR, self.drone_rect)
        
        # Draw direction indicator (simple arrow)
        center = self.drone_rect.center
        if self.command.startswith("MOVE_") or self.command == "FOLLOW":
            end_point = (
                center[0] + (self.target_x - self.drone_x) * 2,
                center[1] + (self.target_y - self.drone_y) * 2
            )
            pygame.draw.line(surface, (255, 255, 0), center, end_point, 2)
            
        # Display status text
        font = pygame.font.SysFont('Arial', 16)
        info_text = f"Mode: {self.mode.upper()} | Command: {self.command}"
        text_surface = font.render(info_text, True, TEXT_COLOR)
        surface.blit(text_surface, (10, self.sim_rect.bottom - 30))
        
        speed_text = f"Speed: {self.speed} | Altitude: {abs(self.position[1])*10:.1f}m"
        speed_surface = font.render(speed_text, True, TEXT_COLOR)
        surface.blit(speed_surface, (10, self.sim_rect.bottom - 10))

# The 3D integration needs to be handled differently to avoid the import error
def initialize_3d_environment(drone_controller):
    """Initializes the 3D environment for the drone simulation"""
    from ursina import Ursina, Entity, EditorCamera, DirectionalLight, AmbientLight, Vec3, color, application
    
    app = Ursina()
    
    # Set up the camera
    camera = EditorCamera()
    camera.position = (3, 2, 3)
    camera.rotation = (30, -45, 0)
    camera.fov = 60
    
    # Create a ground plane with grid
    ground = Entity(
        model='plane',
        scale=(GRID_SIZE * GRID_COUNT, 1, GRID_SIZE * GRID_COUNT),
        color=color.rgb(50, 50, 60),
        texture='white_cube',
        texture_scale=(GRID_COUNT, GRID_COUNT),
        collider='box'
    )
    
    # Add grid lines
    for i in range(-GRID_COUNT//2, GRID_COUNT//2 + 1):
        # X axis lines
        Entity(model='cube', 
               scale=(GRID_SIZE * GRID_COUNT, 0.01, 0.01), 
               position=(0, 0, i * GRID_SIZE), 
               color=color.rgba(200, 200, 200, 0.3))
        # Z axis lines
        Entity(model='cube', 
               scale=(0.01, 0.01, GRID_SIZE * GRID_COUNT), 
               position=(i * GRID_SIZE, 0, 0), 
               color=color.rgba(200, 200, 200, 0.3))
    
    # Create a drone entity (using a simple shape if model is not available)
    try:
        # Try to load the model, fall back to basic cube if model isn't available
        drone_model = Entity(
            model='cube',  # Using cube instead of potentially missing model
            scale=(0.5, 0.2, 0.5),
            color=color.rgb(DRONE_COLOR[0], DRONE_COLOR[1], DRONE_COLOR[2]),
            position=(0, 0, 0)
        )
        
        # Add propellers
        propeller_positions = [
            Vec3(0.25, 0.1, 0.25),  # Front-right
            Vec3(-0.25, 0.1, 0.25),  # Front-left
            Vec3(0.25, 0.1, -0.25),  # Back-right
            Vec3(-0.25, 0.1, -0.25)  # Back-left
        ]
        
        propellers = []
        for pos in propeller_positions:
            propeller = Entity(
                parent=drone_model,
                model='cube',  # Use cube instead of potentially missing cylinder
                scale=(0.1, 0.02, 0.1),
                position=pos,
                color=color.rgb(80, 80, 80)
            )
            propellers.append(propeller)
        
        drone_controller.ursina_drone = drone_model
        
    except Exception as e:
        print(f"Error creating drone model: {e}")
        # Create a simple fallback drone
        drone_model = Entity(
            model='cube',
            scale=0.5,
            color=color.rgb(DRONE_COLOR[0], DRONE_COLOR[1], DRONE_COLOR[2]),
            position=(0, 0, 0)
        )
        drone_controller.ursina_drone = drone_model
    
    # Add directional light
    DirectionalLight(y=2, z=3, rotation=(30, -45, 0))
    
    # Add ambient light
    AmbientLight(color=color.rgba(0.6, 0.6, 0.6, 0.6))
    
    return app, application

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Gesture Controlled Drone Simulator")
    clock = pygame.time.Clock()
    
    # Calculate areas for video feed and simulation
    video_area_width = int(SCREEN_WIDTH * VIDEO_AREA_WIDTH_RATIO)
    sim_area_width = SCREEN_WIDTH - video_area_width
    
    video_area_rect = pygame.Rect(sim_area_width, 0, video_area_width, SCREEN_HEIGHT)
    sim_area_rect = pygame.Rect(0, 0, sim_area_width, SCREEN_HEIGHT)
    
    # Initialize the drone controller
    drone_controller = DroneController(sim_area_rect)
    
    # Initialize webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    # Check if webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize the gesture recognizer
    gesture_recognizer = GestureRecognizer()
    
    # Flag to control if 3D simulation is running
    ursina_initialized = False
    ursina_app = None
    ursina_application = None
    
    # Handle 3D environment initialization in main thread if enabled
    if ENABLE_3D:
        try:
            print("Initializing 3D environment...")
            ursina_app, ursina_application = initialize_3d_environment(drone_controller)
            ursina_initialized = True
        except Exception as e:
            print(f"Error initializing 3D environment: {e}")
            print("Falling back to 2D mode...")
            global ENABLE_2D
            ENABLE_2D = True
    
    running = True
    last_gesture_time = time.time()
    
    try:
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Mirror the frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with the gesture recognizer
            gesture_recognizer.process_frame(frame_rgb)
            
            # Get landmarks and recognize gesture
            landmarks = gesture_recognizer.get_landmarks()
            
            # Draw the landmarks on the frame
            frame_with_landmarks = gesture_recognizer.draw_landmarks(frame.copy())
            
            # Check if a new gesture is detected
            current_time = time.time()
            if landmarks and current_time - last_gesture_time > 0.2:  # Throttle gesture updates
                gesture, vector, intensity = gesture_recognizer.recognize(landmarks)
                stable_gesture, stable_vector, stable_intensity = gesture_recognizer.get_stable_gesture(gesture, vector, intensity)
                if stable_gesture != "UNKNOWN":
                    drone_controller.update_command(stable_gesture, stable_vector, stable_intensity)
                    last_gesture_time = current_time
            
            # Update drone position
            drone_controller.update()
            
            # Clear the screen
            screen.fill(BG_COLOR)
            
            # Draw the simulation area background
            pygame.draw.rect(screen, SIM_BG_COLOR, sim_area_rect)
            
            # Draw the video area background
            pygame.draw.rect(screen, VIDEO_BG_COLOR, video_area_rect)
            
            # Draw the drone in 2D mode
            if ENABLE_2D:
                drone_controller.draw(screen)
            
            # Convert the OpenCV frame to a Pygame surface
            frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Scale the frame to fit the video area if needed
            frame_height = int(video_area_width * frame_rgb.shape[0] / frame_rgb.shape[1])
            if frame_height > SCREEN_HEIGHT:
                frame_height = SCREEN_HEIGHT
                frame_width = int(frame_height * frame_rgb.shape[1] / frame_rgb.shape[0])
            else:
                frame_width = video_area_width
            
            frame_surface = pygame.transform.scale(frame_surface, (frame_width, frame_height))
            
            # Position the frame in the video area
            frame_pos_x = sim_area_width + (video_area_width - frame_width) // 2
            frame_pos_y = (SCREEN_HEIGHT - frame_height) // 2
            
            # Draw the frame on the screen
            screen.blit(frame_surface, (frame_pos_x, frame_pos_y))
            
            # Draw instructions
            font = pygame.font.SysFont('Arial', 20)
            instruction_y = frame_pos_y + frame_height + 10
            
            instructions = [
                "Gesture Controls:",
                "- FIST: Stop the drone",
                "- OPEN_PALM: Hover in place",
                "- POINTING: Move in the direction",
                "- PINCH: Follow hand position",
                "- PEACE: Switch to normal mode",
                "- THREE FINGERS: Switch to rotation mode",
                "- ROCK_ON: Switch to height mode",
                "- THUMBS_UP/DOWN: Increase/decrease speed"
            ]
            
            for instruction in instructions:
                text_surface = font.render(instruction, True, TEXT_COLOR)
                screen.blit(text_surface, (frame_pos_x, instruction_y))
                instruction_y += 25
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(30)
            
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up resources
        gesture_recognizer.close()
        cap.release()
        pygame.quit()
        
        # Properly close Ursina
        if ursina_initialized and ursina_application:
            try:
                # Use the proper method to exit Ursina
                ursina_application.quit()
            except Exception as e:
                print(f"Error while closing Ursina: {e}")

if __name__ == "__main__":
    # Run the main function directly in the main thread
    main()
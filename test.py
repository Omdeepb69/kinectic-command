import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import math
import time
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Configuration
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

# 3D Mode settings
ENABLE_3D = False
ENABLE_2D = True
GRID_SIZE = 20
GRID_COUNT = 10

# Drone trail settings
ENABLE_TRAIL = True
TRAIL_LENGTH = 50
TRAIL_COLOR = (80, 130, 200, 150)

# MediaPipe Hands setup
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
            
        # Get coordinates for key landmarks
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

        # Calculate palm center
        palm_center = np.mean([index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)
        
        # Calculate palm size
        palm_size = np.linalg.norm(wrist - middle_mcp)
        if palm_size < 1e-6: palm_size = 1e-6

        # Get z-coordinate for 3D information
        z_value = landmarks[mp_hands.HandLandmark.WRIST]['z']
        hand_depth = z_value

        # Calculate hand orientation
        palm_normal = np.cross(
            np.append(middle_mcp - index_mcp, 0), 
            np.append(ring_mcp - index_mcp, 0)
        )
        palm_facing_camera = palm_normal[2] < 0

        # Check curl state of each finger
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

        # Special detection for pinch gesture
        pinch_distance = np.linalg.norm(thumb_tip - index_tip) / palm_size
        is_pinching = pinch_distance < 0.15

        # Get pointing direction if index finger is extended
        pointing_vector = None
        pointing_intensity = 0.0
        
        if fingers_extended[1]:
            pointing_vector = index_tip - index_pip
            pointing_magnitude = np.linalg.norm(pointing_vector)
            
            finger_straightness = 1.0 - min(1.0, self._vector_angle(index_tip - index_pip, index_pip - index_mcp) / 40.0)
            pointing_intensity = pointing_magnitude * finger_straightness
            
            if pointing_magnitude > 1e-6:
                pointing_vector /= pointing_magnitude

        # Gesture recognition logic
        
        # FIST - all fingers curled
        if all(fingers_curled):
            return "FIST", None, 0.0
            
        # OPEN_PALM - all fingers extended
        elif all(fingers_extended):
            return "OPEN_PALM", None, 0.0
            
        # THUMBS_UP - only thumb extended, palm not facing camera
        elif fingers_extended[0] and not any(fingers_extended[1:]) and not palm_facing_camera:
            return "THUMBS_UP", None, 0.0
            
        # THUMBS_DOWN - only thumb extended, palm facing camera
        elif fingers_extended[0] and not any(fingers_extended[1:]) and palm_facing_camera:
            return "THUMBS_DOWN", None, 0.0
            
        # PINCH - thumb and index finger close together
        elif is_pinching:
            pinch_center = (thumb_tip + index_tip) / 2
            pinch_vector = pinch_center - palm_center
            pinch_magnitude = np.linalg.norm(pinch_vector)
            if pinch_magnitude > 1e-6:
                pinch_vector /= pinch_magnitude
            return "PINCH", pinch_vector, pinch_distance
            
        # POINTING - index finger extended, all others curled 
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
            
        # PEACE / VICTORY - index and middle fingers extended
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not any(fingers_extended[3:]):
            return "PEACE", None, 0.0
            
        # THREE - index, middle, and ring fingers extended
        elif fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[0] and not fingers_extended[4]:
            return "THREE", None, 0.0
            
        # ROCK_ON - index and pinky extended, others curled
        elif fingers_extended[1] and fingers_extended[4] and not any(fingers_extended[0:1] + fingers_extended[2:4]):
            return "ROCK_ON", None, 0.0
        
        # CALL_ME - thumb, pinky extended, others curled
        elif fingers_extended[0] and fingers_extended[4] and not any(fingers_extended[1:4]):
            return "CALL_ME", None, 0.0
            
        # OK_SIGN - thumb and index form a circle, others extended
        elif is_pinching and all(fingers_extended[2:]):
            return "OK_SIGN", None, 0.0
            
        # Default - unknown gesture
        return "UNKNOWN", None, 0.0

    def get_stable_gesture(self, current_gesture, vector, intensity):
        # Add current gesture to history
        self.gesture_history.append((current_gesture, vector, intensity))
        
        # Keep only the last N gestures
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
            
        # Count occurrences of each gesture in history
        gesture_counts = {}
        for g, _, _ in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
        # Find the most common gesture
        if gesture_counts:
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            # If it appears in majority of frames, return it
            if most_common[1] >= self.history_size / 2:
                self.current_gesture = most_common[0]
                # Find the most recent occurrence of this gesture to get its vector
                for g, v, i in reversed(self.gesture_history):
                    if g == most_common[0]:
                        return g, v, i
        
        # Default to current gesture if no stability
        self.current_gesture = current_gesture
        return current_gesture, vector, intensity

    def draw_landmarks(self, image):
        if self.results and self.results.multi_hand_landmarks:
            # Draw hand landmarks with custom visualization
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw standard connections first
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            # Enhanced visualization based on detected gesture
            landmarks = self.get_landmarks()
            if landmarks:
                # Draw gesture-specific highlights
                if self.current_gesture == "FIST":
                    # Highlight wrist in red
                    cv2.circle(image, 
                            (int(landmarks[mp_hands.HandLandmark.WRIST]['x'] * image.shape[1]),
                             int(landmarks[mp_hands.HandLandmark.WRIST]['y'] * image.shape[0])),
                            15, (0, 0, 255), -1)
                
                elif self.current_gesture == "OPEN_PALM":
                    # Highlight fingertips in yellow
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
                    # Draw arrow in direction of pointing
                    index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y'] * image.shape[0]))
                    
                    # Direction vector based on gesture
                    dx, dy = 0, 0
                    if self.current_gesture == "POINTING_UP": dy = -50
                    elif self.current_gesture == "POINTING_DOWN": dy = 50
                    elif self.current_gesture == "POINTING_LEFT": dx = -50
                    elif self.current_gesture == "POINTING_RIGHT": dx = 50
                    
                    # Draw direction arrow
                    cv2.arrowedLine(image, index_tip, 
                                   (index_tip[0] + dx, index_tip[1] + dy),
                                   (255, 0, 0), 3)
                
                elif self.current_gesture == "PINCH":
                    # Highlight pinch points
                    thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.THUMB_TIP]['y'] * image.shape[0]))
                    index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'] * image.shape[1]),
                                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y'] * image.shape[0]))
                    
                    # Draw circle at pinch center
                    pinch_center = ((thumb_tip[0] + index_tip[0])//2, (thumb_tip[1] + index_tip[1])//2)
                    cv2.circle(image, pinch_center, 10, (255, 0, 255), -1)
                
                # Draw palm center for reference
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
                
            # Add gesture text with larger font and background
            cv2.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.putText(image, f"Gesture: {self.current_gesture}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
        return image

    def close(self):
        self.hands.close()


class Drone3D:
    def __init__(self):
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.size = 1.0
        self.color = (0.0, 0.7, 1.0)
        self.trail = []
        self.trail_max_length = TRAIL_LENGTH
        
    def update_position(self, new_position):
        self.position = new_position
        if ENABLE_TRAIL:
            self.trail.append(list(self.position))
            if len(self.trail) > self.trail_max_length:
                self.trail.pop(0)
    
    def update_rotation(self, new_rotation):
        self.rotation = new_rotation
    
    def draw(self):
        glPushMatrix()
        
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # Draw drone body (central sphere)
        glColor3f(*self.color)
        quad = gluNewQuadric()
        gluSphere(quad, self.size * 0.5, 16, 16)
        
        # Draw rotors (4 smaller spheres)
        rotor_positions = [
            [self.size, 0, self.size],
            [self.size, 0, -self.size],
            [-self.size, 0, self.size],
            [-self.size, 0, -self.size]
        ]
        
        glColor3f(0.8, 0.8, 0.8)
        for pos in rotor_positions:
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            gluSphere(quad, self.size * 0.25, 8, 8)
            glPopMatrix()
            
            glPushMatrix()
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            glPopMatrix()
        
        glPopMatrix()
        
        # Draw trail
        if ENABLE_TRAIL and len(self.trail) > 1:
            glPushMatrix()
            glBegin(GL_LINE_STRIP)
            
            for i, pos in enumerate(self.trail):
                alpha = i / len(self.trail)
                glColor4f(TRAIL_COLOR[0]/255, TRAIL_COLOR[1]/255, TRAIL_COLOR[2]/255, alpha * TRAIL_COLOR[3]/255)
                glVertex3f(pos[0], pos[1], pos[2])
            
            glEnd()
            glPopMatrix()


class Scene3D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.drone = Drone3D()
        self.setup_gl()
        
    def setup_gl(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        gluLookAt(0, 5, 15, 0, 0, 0, 0, 1, 0)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
    def resize(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width / height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def draw_grid(self):
        glBegin(GL_LINES)
        
        # Ground grid
        glColor4f(0.5, 0.5, 0.5, 0.5)
        grid_size = GRID_SIZE
        grid_count = GRID_COUNT
        
        for i in range(-grid_count, grid_count + 1):
            # X lines
            glVertex3f(i * grid_size / grid_count, 0, -grid_size)
            glVertex3f(i * grid_size / grid_count, 0, grid_size)
            
            # Z lines
            glVertex3f(-grid_size, 0, i * grid_size / grid_count)
            glVertex3f(grid_size, 0, i * grid_size / grid_count)
        
        # Coordinate axes
        # X axis - red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(grid_size, 0, 0)
        
        # Y axis - green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, grid_size, 0)
        
        # Z axis - blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, grid_size)
        
        glEnd()
        
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluLookAt(0, 5, 15, 0, 0, 0, 0, 1, 0)
        
        self.draw_grid()
        self.drone.draw()
        
    def update_drone(self, position, rotation):
        self.drone.update_position(position)
        self.drone.update_rotation(rotation)


class DroneSimulator:
    def __init__(self, screen_width, screen_height, sim_area_rect):
        self.sim_rect = sim_area_rect
        
        # 2D position (for legacy mode)
        self.drone_x = self.sim_rect.centerx
        self.drone_y = self.sim_rect.centery
        self.drone_rect = pygame.Rect(0, 0, DRONE_SIZE, DRONE_SIZE)
        self.drone_rect.center = (self.drone_x, self.drone_y)
        
        # 3D position and orientation
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.target_rotation = [0.0, 0.0, 0.0]
        
        # Movement parameters
        self.speed = DRONE_SPEED
        self.vertical_speed = DRONE_VERTICAL_SPEED
        self.rotation_speed = DRONE_ROTATION_SPEED
        self.target_x = self.drone_x
        self.target_y = self.drone_y
        self.target_z = 0.0
        
        # Control parameters
        self.command = "STOP"
        self.prev_command = "STOP"
        self.mode = "normal"
        
        # For 3D visualization
        if ENABLE_3D:
            self.scene = Scene3D(sim_area_rect.width, sim_area_rect.height)
        
        # Drone trail for visualization
        self.trail = []
        self.trail_max_length = TRAIL_LENGTH
        
    def update_command(self, gesture, vector=None, intensity=0.0):
        self.prev_command = self.command
        
        # Mode switching based on special gestures
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
        
        # Normal mode commands
        if self.mode == "normal":
            if gesture == "FIST":
                self.command = "STOP"
            elif gesture == "OPEN_PALM":
                self.command = "HOVER"
            elif "POINTING" in gesture:
                self.command = gesture.replace("POINTING_", "MOVE_")
                if vector is not None and gesture != "POINTING":
                    # Set target position based on pointing direction
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
                    
                    # Update 3D targets
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
                    
                    # Update 3D targets
                    self.target_position = [
                        (self.target_x - center_x) / 100.0,
                        0,
                        (self.target_y - center_y) / 100.0
                    ]
        
        # Rotation mode commands
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
        
        # Height mode commands
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
        # Move drone towards target position
        dx = self.target_x - self.drone_x
        dy = self.target_y - self.drone_y
        dz = self.target_z - self.position[1]
        
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 1:
            # Normalize and scale by speed
            move_x = dx * min(1, self.speed / distance)
            move_y = dy * min(1, self.speed / distance)
            
            # Update position
            self.drone_x += move_x
            self.drone_y += move_y
            
            # Keep within bounds
            self.drone_x = max(self.sim_rect.left, min(self.sim_rect.right, self.drone_x))
            self.drone_y = max(self.sim_rect.top, min(self.sim_rect.bottom, self.drone_y))
            
            # Update rect
            self.drone_rect.center = (self.drone_x, self.drone_y)
        
        # Update 3D position
        target_position = [
            (self.drone_x - self.sim_rect.centerx) / 100.0,
            self.target_z,
            (self.drone_y - self.sim_rect.centery) / 100.0
        ]
        
        # Smooth interpolation for 3D position
        self.position[0] += (target_position[0] - self.position[0]) * 0.2
        self.position[1] += (target_position[1] - self.position[1]) * 0.2
        self.position[2] += (target_position[2] - self.position[2]) * 0.2
        
        # Smooth interpolation for rotation
        for i in range(3):
            self.rotation[i] += (self.target_rotation[i] - self.rotation[i]) * 0.1
        
        # Update trail
        if ENABLE_TRAIL:
            self.trail.append((self.drone_x, self.drone_y))
            if len(self.trail) > self.trail_max_length:
                self.trail.pop(0)
        
        # Update 3D scene
        if ENABLE_3D:
            self.scene.update_drone(self.position, self.rotation)
    
    def draw(self, surface):
        # Draw 3D scene
        if ENABLE_3D:
            self.scene.render()
        else:
            # Legacy 2D rendering
            # Draw drone trail
            if ENABLE_TRAIL and len(self.trail) > 1:
                for i in range(1, len(self.trail)):
                    alpha = int(255 * i / len(self.trail))
                    color = (*TRAIL_COLOR[:3], alpha)
                    pygame.draw.line(surface, color, self.trail[i-1], self.trail[i], 2)
            
            # Draw drone
            pygame.draw.circle(surface, DRONE_COLOR, self.drone_rect.center, DRONE_SIZE//2)
            
            # Draw direction indicator
            angle = math.atan2(self.target_y - self.drone_y, self.target_x - self.drone_x)
            end_x = self.drone_x + math.cos(angle) * DRONE_SIZE
            end_y = self.drone_y + math.sin(angle) * DRONE_SIZE
            pygame.draw.line(surface, (255, 255, 255), self.drone_rect.center, (end_x, end_y), 3)
        
        # Draw command text
        font = pygame.font.Font(None, 36)
        mode_text = font.render(f"Mode: {self.mode.upper()}", True, TEXT_COLOR)
        cmd_text = font.render(f"Command: {self.command}", True, TEXT_COLOR)
        
        surface.blit(mode_text, (self.sim_rect.left + 10, self.sim_rect.top + 10))
        surface.blit(cmd_text, (self.sim_rect.left + 10, self.sim_rect.top + 50))


def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Gesture Control")
    clock = pygame.time.Clock()
    
    # Initialize areas
    sim_area_width = int(SCREEN_WIDTH * SIM_AREA_WIDTH_RATIO)
    video_area_width = SCREEN_WIDTH - sim_area_width
    
    sim_area_rect = pygame.Rect(0, 0, sim_area_width, SCREEN_HEIGHT)
    video_area_rect = pygame.Rect(sim_area_width, 0, video_area_width, SCREEN_HEIGHT)
    
    # Initialize simulator
    simulator = DroneSimulator(SCREEN_WIDTH, SCREEN_HEIGHT, sim_area_rect)
    
    # Initialize webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        pygame.quit()
        sys.exit()
    
    # Initialize gesture recognizer
    recognizer = GestureRecognizer()
    
    # Font setup
    font = pygame.font.Font(None, 30)
    
    running = True
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Read webcam frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand gestures
        recognizer.process_frame(frame_rgb)
        landmarks = recognizer.get_landmarks()
        
        if landmarks:
            gesture, vector, intensity = recognizer.recognize(landmarks)
            stable_gesture, stable_vector, stable_intensity = recognizer.get_stable_gesture(gesture, vector, intensity)
            simulator.update_command(stable_gesture, stable_vector, stable_intensity)
        else:
            stable_gesture = "NO_HAND"
            simulator.update_command("NO_HAND")
        
        # Update simulator
        simulator.update()
        
        # Draw gesture visualization on frame
        frame = recognizer.draw_landmarks(frame)
        
        # Convert frame to Pygame surface
        frame = cv2.resize(frame, (video_area_rect.width, video_area_rect.height))
        frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
        
        # Fill screen with background colors
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, SIM_BG_COLOR, sim_area_rect)
        pygame.draw.rect(screen, VIDEO_BG_COLOR, video_area_rect)
        
        # Draw simulator
        simulator.draw(screen)
        
        # Draw webcam frame
        screen.blit(frame_surface, video_area_rect.topleft)
        
        # Add gesture label
        gesture_text = font.render(f"Gesture: {stable_gesture}", True, GESTURE_TEXT_COLOR)
        screen.blit(gesture_text, (video_area_rect.left + 10, video_area_rect.bottom - 40))
        
        # Add instruction text
        instructions = [
            "Controls:",
            "FIST: Stop",
            "OPEN_PALM: Hover",
            "POINTING: Move direction",
            "PINCH: Follow hand",
            "PEACE: Normal mode",
            "THREE: Rotation mode",
            "ROCK_ON: Height mode",
            "THUMBS_UP/DOWN: Speed up/down"
        ]
        
        for i, text in enumerate(instructions):
            help_text = font.render(text, True, TEXT_COLOR)
            screen.blit(help_text, (video_area_rect.left + 10, video_area_rect.top + 10 + i * 30))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    recognizer.close()
    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
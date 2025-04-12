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

# --- Configuration ---
WEBCAM_INDEX = 0
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SIM_AREA_WIDTH_RATIO = 0.6  # Percentage of screen width for simulation
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
ENABLE_3D = True
GRID_SIZE = 20
GRID_COUNT = 10

# Drone trail settings
ENABLE_TRAIL = True
TRAIL_LENGTH = 50
TRAIL_COLOR = (80, 130, 200, 150)  # RGBA with alpha

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Gesture Recognition Logic ---
class GestureRecognizer:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            model_complexity=1,  # Increased from 0 to 1 for better accuracy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1)
        self.results = None
        self.image_height = 0
        self.image_width = 0
        self.smoothed_landmarks = None
        self.alpha = 0.7  # Smoothing factor (0-1), higher = more smoothing
        self.gesture_history = []
        self.history_size = 5  # Number of frames to keep for gesture stability
        
    def process_frame(self, image_rgb):
        self.image_height, self.image_width = image_rgb.shape[:2]
        self.results = self.hands.process(image_rgb)
        
        # Apply landmark smoothing
        if self.results and self.results.multi_hand_landmarks:
            new_landmarks = self.results.multi_hand_landmarks[0].landmark
            
            if self.smoothed_landmarks is None:
                # First detection, initialize smoothed landmarks
                self.smoothed_landmarks = [
                    {'x': l.x, 'y': l.y, 'z': l.z} for l in new_landmarks
                    for l in new_landmarks
                ]
            else:
                # Apply exponential smoothing to each landmark
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
            # Return landmarks for the first detected hand
            return self.results.multi_hand_landmarks[0].landmark
        return None

    def _vector_angle(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)
    
    def _finger_is_curved(self, finger_landmarks, threshold=0.4):
        """Check if a finger is curved (bent) based on its landmarks"""
        # Extract the 3 points of the finger (tip, mid, base)
        tip = np.array([finger_landmarks[0]['x'], finger_landmarks[0]['y']])
        mid = np.array([finger_landmarks[1]['x'], finger_landmarks[1]['y']])
        base = np.array([finger_landmarks[2]['x'], finger_landmarks[2]['y']])
        
        # Calculate vectors
        v1 = mid - base
        v2 = tip - mid
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return False
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # Convert to degrees
        angle_deg = np.degrees(angle)
        
        # If angle is greater than threshold, finger is curved
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
        
        # Calculate palm size (distance from wrist to middle MCP)
        palm_size = np.linalg.norm(wrist - middle_mcp)
        if palm_size < 1e-6: palm_size = 1e-6  # Avoid division by zero

        # Get z-coordinate for 3D information
        z_value = landmarks[mp_hands.HandLandmark.WRIST]['z']
        hand_depth = z_value  # Relative depth, negative is closer to camera

        # Calculate hand orientation (important for gesture context)
        palm_normal = np.cross(
            np.append(middle_mcp - index_mcp, 0), 
            np.append(ring_mcp - index_mcp, 0)
        )
        palm_facing_camera = palm_normal[2] < 0  # Negative z means palm facing camera

        # Check curl state of each finger
        # Using more sophisticated approach with vectors and angles
        fingers = [
            [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], landmarks[mp_hands.HandLandmark.RING_FINGER_PIP], landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]],
            [landmarks[mp_hands.HandLandmark.PINKY_TIP], landmarks[mp_hands.HandLandmark.PINKY_PIP], landmarks[mp_hands.HandLandmark.PINKY_MCP]]
        ]
        
        finger_curled = [self._finger_is_curved(finger) for finger in fingers]
        
        # Thumb is special - check distance to index finger base
        thumb_curled = np.linalg.norm(thumb_tip - index_mcp) < np.linalg.norm(wrist - index_mcp) * 0.4
        
        # Combine curl states
        fingers_curled = [thumb_curled] + finger_curled
        fingers_extended = [not curled for curled in fingers_curled]

        # Special detection for pinch gesture
        pinch_distance = np.linalg.norm(thumb_tip - index_tip) / palm_size
        is_pinching = pinch_distance < 0.15

        # Get pointing direction if index finger is extended
        pointing_vector = None
        pointing_intensity = 0.0
        
        if fingers_extended[1]:  # Index finger extended
            # Use multiple points for better direction vector
            pointing_vector = index_tip - index_pip
            pointing_magnitude = np.linalg.norm(pointing_vector)
            
            # Calculate "intensity" based on how straight the finger is
            finger_straightness = 1.0 - min(1.0, self._vector_angle(index_tip - index_pip, index_pip - index_mcp) / 40.0)
            pointing_intensity = pointing_magnitude * finger_straightness
            
            # Normalize vector
            if pointing_magnitude > 1e-6:
                pointing_vector /= pointing_magnitude

        # Gesture recognition logic - more robust
        
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
            # Calculate pinch center
            pinch_center = (thumb_tip + index_tip) / 2
            # Direction from palm center to pinch center
            pinch_vector = pinch_center - palm_center
            pinch_magnitude = np.linalg.norm(pinch_vector)
            if pinch_magnitude > 1e-6:
                pinch_vector /= pinch_magnitude
            return "PINCH", pinch_vector, pinch_distance
            
        # POINTING - index finger extended, all others curled 
        elif fingers_extended[1] and not any([fingers_extended[0]] + fingers_extended[2:]):
            if pointing_vector is not None:
                # Determine direction
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
                # Find the most recent occurrence of this gesture to get its vector
                for g, v, i in reversed(self.gesture_history):
                    if g == most_common[0]:
                        return g, v, i
                        
        # Default to current gesture if no stability
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
                    
            # Add some extra visualization for better feedback
            landmarks = self.get_landmarks()
            if landmarks:
                # Draw palm center
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
                
                cv2.circle(image, (palm_x, palm_y), 10, (255, 0, 255), -1)
                
                # Draw index fingertip with special emphasis if pointing
                index_tip_x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['x'] * image.shape[1])
                index_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]['y'] * image.shape[0])
                cv2.circle(image, (index_tip_x, index_tip_y), 8, (0, 255, 255), -1)
                
        return image

    def close(self):
        self.hands.close()


# --- 3D Visualization ---
class Drone3D:
    def __init__(self):
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]  # pitch, yaw, roll
        self.size = 1.0
        self.color = (0.0, 0.7, 1.0)  # RGB, 0-1 scale for OpenGL
        self.trail = []
        self.trail_max_length = TRAIL_LENGTH
        
    def update_position(self, new_position):
        self.position = new_position
        # Update trail
        if ENABLE_TRAIL:
            self.trail.append(list(self.position))
            if len(self.trail) > self.trail_max_length:
                self.trail.pop(0)
    
    def update_rotation(self, new_rotation):
        self.rotation = new_rotation
    
    def draw(self):
        # Save matrix state
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glRotatef(self.rotation[0], 1, 0, 0)  # Pitch (X)
        glRotatef(self.rotation[1], 0, 1, 0)  # Yaw (Y)
        glRotatef(self.rotation[2], 0, 0, 1)  # Roll (Z)
        
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
        
        glColor3f(0.8, 0.8, 0.8)  # Light gray for rotors
        for pos in rotor_positions:
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            gluSphere(quad, self.size * 0.25, 8, 8)
            glPopMatrix()
            
            # Draw connecting arms
            glPushMatrix()
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            glPopMatrix()
        
        # Restore matrix state
        glPopMatrix()
        
        # Draw trail
        if ENABLE_TRAIL and len(self.trail) > 1:
            glPushMatrix()
            glBegin(GL_LINE_STRIP)
            
            # Gradient trail from transparent to solid
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
        # Setup perspective
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Set camera position
        gluLookAt(0, 5, 15,  # Eye position
                  0, 0, 0,    # Look at position
                  0, 1, 0)    # Up vector
        
        # Enable features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Nice lines for grid
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
        # Draw coordinate grid
        glBegin(GL_LINES)
        
        # Ground grid
        glColor4f(0.5, 0.5, 0.5, 0.5)  # Gray with transparency
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
        # Clear screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera position
        gluLookAt(0, 5, 15,  # Eye position
                  0, 0, 0,    # Look at position
                  0, 1, 0)    # Up vector
        
        # Draw scene elements
        self.draw_grid()
        self.drone.draw()
        
    def update_drone(self, position, rotation):
        self.drone.update_position(position)
        self.drone.update_rotation(rotation)


# --- Simulation Logic ---
class DroneSimulator:
    def __init__(self, screen_width, screen_height, sim_area_rect):
        self.sim_rect = sim_area_rect
        
        # 2D position (for legacy mode)
        self.drone_x = self.sim_rect.centerx
        self.drone_y = self.sim_rect.centery
        self.drone_rect = pygame.Rect(0, 0, DRONE_SIZE, DRONE_SIZE)
        self.drone_rect.center = (self.drone_x, self.drone_y)
        
        # 3D position and orientation
        self.position = [0.0, 0.0, 0.0]  # x, y, z in OpenGL coordinates
        self.rotation = [0.0, 0.0, 0.0]  # pitch, yaw, roll in degrees
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
        self.mode = "normal"  # normal, rotation, height
        
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
            elif gesture == "POINTING_LEFT":
                self.command = "LEFT"
            elif gesture == "POINTING_RIGHT":
                self.command = "RIGHT"
            elif gesture == "POINTING_UP":
                self.command = "FORWARD"
            elif gesture == "POINTING_DOWN":
                self.command = "BACKWARD"
            elif gesture == "PINCH":
                self.command = "RETURN_HOME"
            elif gesture == "OK_SIGN":
                self.command = "SPIN"
            else:
                # Keep last command if gesture not recognized
                pass
                
        # Rotation mode commands
        elif self.mode == "rotation":
            if gesture == "POINTING_LEFT":
                self.command = "ROTATE_LEFT"
            elif gesture == "POINTING_RIGHT":
                self.command = "ROTATE_RIGHT"
            elif gesture == "POINTING_UP":
                self.command = "TILT_UP"
            elif gesture == "POINTING_DOWN":
                self.command = "TILT_DOWN"
            elif gesture == "FIST":
                self.command = "STOP_ROTATION"
            elif gesture == "PINCH":
                self.command = "RESET_ROTATION"
            else:
                # Keep last command if gesture not recognized
                pass
                
        # Height mode commands
        elif self.mode == "height":
            if gesture == "POINTING_UP":
                self.command = "UP"
            elif gesture == "POINTING_DOWN":
                self.command = "DOWN"
            elif gesture == "FIST":
                self.command = "STOP_HEIGHT"
            elif gesture == "PINCH":
                self.command = "RESET_HEIGHT"
            else:
                # Keep last command if gesture not recognized
                pass
    
    def update(self):
        # Update drone position and rotation based on current command
        
        # Handle movement commands
        if self.mode == "normal":
            if self.command == "LEFT":
                if ENABLE_3D:
                    self.position[0] -= self.speed * 0.05
                else:
                    self.target_x -= self.speed
            elif self.command == "RIGHT":
                if ENABLE_3D:
                    self.position[0] += self.speed * 0.05
                else:
                    self.target_x += self.speed
            elif self.command == "FORWARD":
                if ENABLE_3D:
                    self.position[2] -= self.speed * 0.05
                else:
                    self.target_y -= self.speed
            elif self.command == "BACKWARD":
                if ENABLE_3D:
                    self.position[2] += self.speed * 0.05
                else:
                    self.target_y += self.speed
            elif self.command == "RETURN_HOME":
                if ENABLE_3D:
                    self.target_x = 0
                    self.target_y = 0
                    self.target_z = 0
                    
                    # Move towards home position
                    dx = 0 - self.position[0]
                    dy = 0 - self.position[1]
                    dz = 0 - self.position[2]
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance > 0.1:
                        self.position[0] += dx * 0.05
                        self.position[1] += dy * 0.05
                        self.position[2] += dz * 0.05
                else:
                    self.target_x = self.sim_rect.centerx
                    self.target_y = self.sim_rect.centery
            elif self.command == "SPIN":
                self.target_rotation[1] += self.rotation_speed * 2
                if self.target_rotation[1] >= 360:
                    self.target_rotation[1] -= 360
        
        # Handle rotation commands
        elif self.mode == "rotation":
            if self.command == "ROTATE_LEFT":
                self.target_rotation[1] -= self.rotation_speed
                if self.target_rotation[1] < 0:
                    self.target_rotation[1] += 360
            elif self.command == "ROTATE_RIGHT":
                self.target_rotation[1] += self.rotation_speed
                if self.target_rotation[1] >= 360:
                    self.target_rotation[1] -= 360
            elif self.command == "TILT_UP":
                self.target_rotation[0] -= self.rotation_speed
                self.target_rotation[0] = max(-45, self.target_rotation[0])
            elif self.command == "TILT_DOWN":
                self.target_rotation[0] += self.rotation_speed
                self.target_rotation[0] = min(45, self.target_rotation[0])
            elif self.command == "RESET_ROTATION":
                self.target_rotation = [0.0, 0.0, 0.0]
        
        # Handle height commands
        elif self.mode == "height":
            if self.command == "UP":
                if ENABLE_3D:
                    self.position[1] += self.vertical_speed * 0.05
                else:
                    # In 2D, we can show altitude by changing drone size
                    self.drone_rect.width = max(10, self.drone_rect.width - 1)
                    self.drone_rect.height = max(10, self.drone_rect.height - 1)
            elif self.command == "DOWN":
                if ENABLE_3D:
                    self.position[1] -= self.vertical_speed * 0.05
                    # Don't go below ground level
                    self.position[1] = max(0, self.position[1])
                else:
                    # In 2D, we can show altitude by changing drone size
                    self.drone_rect.width = min(DRONE_SIZE * 2, self.drone_rect.width + 1)
                    self.drone_rect.height = min(DRONE_SIZE * 2, self.drone_rect.height + 1)
            elif self.command == "RESET_HEIGHT":
                if ENABLE_3D:
                    self.position[1] = 0
                else:
                    self.drone_rect.width = DRONE_SIZE
                    self.drone_rect.height = DRONE_SIZE
        
        # Smoothly interpolate rotation
        for i in range(3):
            if self.rotation[i] != self.target_rotation[i]:
                diff = self.target_rotation[i] - self.rotation[i]
                # Take shortest path for yaw (around 360)
                if i == 1 and abs(diff) > 180:
                    if diff > 0:
                        diff -= 360
                    else:
                        diff += 360
                # Apply smooth interpolation
                self.rotation[i] += diff * 0.1
        
        # Update 2D position (for legacy mode)
        if not ENABLE_3D:
            # Smoothly move toward target position
            dx = self.target_x - self.drone_x
            dy = self.target_y - self.drone_y
            
            self.drone_x += dx * 0.1
            self.drone_y += dy * 0.1
            
            # Keep within simulation boundaries
            self.drone_x = max(self.sim_rect.left + self.drone_rect.width/2, 
                            min(self.sim_rect.right - self.drone_rect.width/2, self.drone_x))
            self.drone_y = max(self.sim_rect.top + self.drone_rect.height/2, 
                            min(self.sim_rect.bottom - self.drone_rect.height/2, self.drone_y))
            
            self.drone_rect.center = (self.drone_x, self.drone_y)
            
            # Update trail
            if ENABLE_TRAIL:
                self.trail.append((self.drone_x, self.drone_y))
                if len(self.trail) > self.trail_max_length:
                    self.trail.pop(0)
        
        # Update 3D scene
        if ENABLE_3D:
            # Keep position within reasonable bounds
            for i in range(3):
                self.position[i] = max(-GRID_SIZE, min(GRID_SIZE, self.position[i]))
            
            # Update 3D scene
            self.scene.update_drone(self.position, self.rotation)
    
    def render(self, screen):
        if ENABLE_3D:
            # Render the 3D scene
            self.scene.render()
        else:
            # 2D rendering
            if screen is not None:
                pygame.draw.rect(screen, SIM_BG_COLOR, self.sim_rect)
                
                # Draw trail
                if ENABLE_TRAIL and len(self.trail) > 1:
                    # Create gradient colors for trail
                    for i in range(len(self.trail) - 1):
                        alpha = int(255 * i / len(self.trail))
                        color = (TRAIL_COLOR[0], TRAIL_COLOR[1], TRAIL_COLOR[2], alpha)
                        pygame.draw.line(screen, color, self.trail[i], self.trail[i+1], 2)
                
                # Draw drone
                pygame.draw.rect(screen, DRONE_COLOR, self.drone_rect)
                
                # Draw command visualization
                self.draw_command_visualization(screen)
    
    def draw_command_visualization(self, screen):
        # Draw a visual indicator of current command
        text_x = self.sim_rect.left + 10
        text_y = self.sim_rect.top + 10
        font = pygame.font.SysFont(None, 24)
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        mode_surf = font.render(mode_text, True, TEXT_COLOR)
        screen.blit(mode_surf, (text_x, text_y))
        
        # Command indicator
        cmd_text = f"Command: {self.command}"
        cmd_surf = font.render(cmd_text, True, GESTURE_TEXT_COLOR)
        screen.blit(cmd_surf, (text_x, text_y + 25))
        
        # Speed indicator
        speed_text = f"Speed: {self.speed}"
        speed_surf = font.render(speed_text, True, TEXT_COLOR)
        screen.blit(speed_surf, (text_x, text_y + 50))
        
        if ENABLE_3D:
            # Position indicator
            pos_text = f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})"
            pos_surf = font.render(pos_text, True, TEXT_COLOR)
            screen.blit(pos_surf, (text_x, text_y + 75))
            
            # Rotation indicator
            rot_text = f"Rotation: ({self.rotation[0]:.1f}, {self.rotation[1]:.1f}, {self.rotation[2]:.1f})"
            rot_surf = font.render(rot_text, True, TEXT_COLOR)
            screen.blit(rot_surf, (text_x, text_y + 100))


# --- Main Application Class ---
class DroneControlApp:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        # Set up the display - main window only
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.OPENGL if ENABLE_3D else 0)
        pygame.display.set_caption("Gesture Controlled Drone Simulator")
        
        # Calculate area dimensions
        self.sim_width = int(self.screen_width * SIM_AREA_WIDTH_RATIO)
        self.video_width = int(self.screen_width * VIDEO_AREA_WIDTH_RATIO)
        
        # Create rectangles for simulation and video areas
        self.sim_rect = pygame.Rect(0, 0, self.sim_width, self.screen_height)
        self.video_rect = pygame.Rect(self.sim_width, 0, self.video_width, self.screen_height)
        
        # Initialize OpenGL
        if ENABLE_3D:
            self.setup_opengl()
            self.fbo = None
            self.texture_id = None
            self.init_fbo()  # Initialize framebuffer object for rendering to texture
        
        # Create simulation
        self.simulator = DroneSimulator(self.screen_width, self.screen_height, self.sim_rect)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize hand gesture recognition
        self.gesture_recognizer = GestureRecognizer()
        
        # Setup PyGame clock
        self.clock = pygame.time.Clock()
        self.running = True
        
        # For FPS display
        self.font = pygame.font.SysFont(None, 24)
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0
        
        # For gesture history display
        self.gesture_history = []
        self.max_gesture_history = 5
        
        # Create pygame surface for 2D rendering (when in 3D mode)
        if ENABLE_3D:
            self.pygame_surface = pygame.Surface((self.screen_width, self.screen_height))
    
    def setup_opengl(self):
        """Initialize OpenGL settings"""
        # Set clear color 
        glClearColor(0.2, 0.2, 0.3, 1.0)
        
        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        
        # Setup viewport and projection
        glViewport(0, 0, self.sim_width, self.screen_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.sim_width / self.screen_height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def init_fbo(self):
        """Initialize framebuffer object for rendering to texture"""
        # This is a placeholder - in a complete implementation, 
        # you would create an OpenGL framebuffer object here
        # for offscreen rendering to a texture
        pass
    
    def render_to_pygame_surface(self):
        """Render OpenGL content to a pygame surface"""
        # In a complete implementation, this would read the pixels from
        # the framebuffer and convert them to a Pygame surface
        # For simplicity, we just clear our pygame surface for now
        if ENABLE_3D:
            self.pygame_surface.fill(BG_COLOR)
    
    def run(self):
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Process webcam frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            self.gesture_recognizer.process_frame(frame_rgb)
            
            # Draw hand landmarks on frame
            frame_with_landmarks = self.gesture_recognizer.draw_landmarks(frame_rgb)
            
            # Get hand landmarks and recognize gesture
            landmarks = self.gesture_recognizer.get_landmarks()
            if landmarks:
                gesture, vector, intensity = self.gesture_recognizer.recognize(landmarks)
                stable_gesture, stable_vector, stable_intensity = self.gesture_recognizer.get_stable_gesture(gesture, vector, intensity)
                
                # Update gesture history for display
                if stable_gesture != "UNKNOWN" and stable_gesture != "NO_HAND":
                    self.gesture_history.insert(0, stable_gesture)
                    if len(self.gesture_history) > self.max_gesture_history:
                        self.gesture_history.pop()
                
                # Update simulator command
                self.simulator.update_command(stable_gesture, stable_vector, stable_intensity)
                
                # Draw gesture info on frame
                y_pos = 30
                cv2.putText(frame_with_landmarks, f"Gesture: {stable_gesture}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                stable_gesture = "NO_HAND"
                cv2.putText(frame_with_landmarks, "No hand detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update simulator
            self.simulator.update()
            
            # Handle rendering differently based on 3D mode
            if ENABLE_3D:
                # First, render the 3D scene (will render to FBO)
                self.simulator.render(None)
                
                # Switch back to default framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                
                # Render OpenGL content to pygame surface
                self.render_to_pygame_surface()
                
                # Now render everything else on the pygame surface
                self.render_2d_ui(frame_with_landmarks)
                
                # Update display
                pygame.display.flip()
            else:
                # For 2D mode, just render directly to screen
                self.screen.fill(BG_COLOR)
                self.simulator.render(self.screen)
                self.render_ui(frame_with_landmarks)
                pygame.display.flip()
            
            # Update FPS counter
            self.frames += 1
            if time.time() - self.last_time > 1.0:
                self.fps = self.frames
                self.frames = 0
                self.last_time = time.time()
            
            # Cap at 60 FPS
            self.clock.tick(60)
    
    def render_2d_ui(self, frame_with_landmarks):
        """Render UI elements in 2D (for 3D mode)"""
        # Use pygame_surface to draw all 2D elements
        self.pygame_surface.fill(BG_COLOR)
        
        # Draw simulation area background
        pygame.draw.rect(self.pygame_surface, SIM_BG_COLOR, self.sim_rect)
        
        # Draw video area background
        pygame.draw.rect(self.pygame_surface, VIDEO_BG_COLOR, self.video_rect)
        
        # Convert frame to pygame surface and display in video area
        surf = pygame.surfarray.make_surface(frame_with_landmarks.swapaxes(0, 1))
        
        # Scale to fit video area
        surf = pygame.transform.scale(surf, (self.video_rect.width, int(self.video_rect.width * frame_with_landmarks.shape[0] / frame_with_landmarks.shape[1])))
        
        # Display video in video area
        video_y = self.video_rect.top + (self.video_rect.height - surf.get_height()) // 2
        self.pygame_surface.blit(surf, (self.video_rect.left, video_y))
        
        # Draw FPS
        fps_text = self.font.render(f"FPS: {self.fps}", True, TEXT_COLOR)
        self.pygame_surface.blit(fps_text, (self.video_rect.left + 10, 10))
        
        # Draw gesture history
        history_y = 50
        history_text = self.font.render("Recent Gestures:", True, TEXT_COLOR)
        self.pygame_surface.blit(history_text, (self.video_rect.left + 10, history_y))
        
        for i, gesture in enumerate(self.gesture_history):
            gesture_text = self.font.render(f"{i+1}. {gesture}", True, GESTURE_TEXT_COLOR)
            self.pygame_surface.blit(gesture_text, (self.video_rect.left + 20, history_y + 30 + i*25))
        
        # Draw control instructions
        instructions = [
            "Controls:",
            "FIST: Stop",
            "OPEN_PALM: Hover",
            "POINTING: Move direction",
            "PEACE: Normal mode",
            "THREE: Rotation mode",
            "ROCK_ON: Height mode",
            "THUMBS_UP/DOWN: Speed up/down",
            "PINCH: Return home"
        ]
        
        instruction_y = self.video_rect.bottom - len(instructions) * 25 - 10
        for i, text in enumerate(instructions):
            instruction_text = self.font.render(text, True, TEXT_COLOR)
            self.pygame_surface.blit(instruction_text, (self.video_rect.left + 10, instruction_y + i*25))
        
        # Blit the pygame surface to the screen
        temp_surface = self.pygame_surface.copy()
        pygame.image.save(temp_surface, "temp.jpg")
        temp_surface = pygame.image.load("temp.jpg")
        self.screen.blit(temp_surface, (0, 0))
    
    def render_ui(self, frame_with_landmarks):
        """Render UI elements for 2D mode"""
        # Draw video area background
        pygame.draw.rect(self.screen, VIDEO_BG_COLOR, self.video_rect)
        
        # Convert frame to pygame surface and display in video area
        surf = pygame.surfarray.make_surface(frame_with_landmarks.swapaxes(0, 1))
        
        # Scale to fit video area
        surf = pygame.transform.scale(surf, (self.video_rect.width, int(self.video_rect.width * frame_with_landmarks.shape[0] / frame_with_landmarks.shape[1])))
        
        # Display video in video area
        video_y = self.video_rect.top + (self.video_rect.height - surf.get_height()) // 2
        self.screen.blit(surf, (self.video_rect.left, video_y))
        
        # Draw FPS
        fps_text = self.font.render(f"FPS: {self.fps}", True, TEXT_COLOR)
        self.screen.blit(fps_text, (self.video_rect.left + 10, 10))
        
        # Draw gesture history
        history_y = 50
        history_text = self.font.render("Recent Gestures:", True, TEXT_COLOR)
        self.screen.blit(history_text, (self.video_rect.left + 10, history_y))
        
        for i, gesture in enumerate(self.gesture_history):
            gesture_text = self.font.render(f"{i+1}. {gesture}", True, GESTURE_TEXT_COLOR)
            self.screen.blit(gesture_text, (self.video_rect.left + 20, history_y + 30 + i*25))
        
        # Draw control instructions
        instructions = [
            "Controls:",
            "FIST: Stop",
            "OPEN_PALM: Hover",
            "POINTING: Move direction",
            "PEACE: Normal mode",
            "THREE: Rotation mode",
            "ROCK_ON: Height mode",
            "THUMBS_UP/DOWN: Speed up/down",
            "PINCH: Return home"
        ]
        
        instruction_y = self.video_rect.bottom - len(instructions) * 25 - 10
        for i, text in enumerate(instructions):
            instruction_text = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(instruction_text, (self.video_rect.left + 10, instruction_y + i*25))
    
    def cleanup(self):
        # Clean up resources
        self.cap.release()
        self.gesture_recognizer.close()
        pygame.quit()


# --- Main entry point ---
if __name__ == "__main__":
    # Create and run application
    app = DroneControlApp()
    try:
        app.run()
    finally:
        app.cleanup()
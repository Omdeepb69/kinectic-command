import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import math
import time

# --- Configuration ---
WEBCAM_INDEX = 0
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
SIM_AREA_WIDTH_RATIO = 0.4 # Percentage of screen width for simulation
VIDEO_AREA_WIDTH_RATIO = 1.0 - SIM_AREA_WIDTH_RATIO

BG_COLOR = (30, 30, 30)
SIM_BG_COLOR = (50, 50, 50)
VIDEO_BG_COLOR = (40, 40, 40)
TEXT_COLOR = (230, 230, 230)
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 0, 255)
GESTURE_TEXT_COLOR = (255, 255, 0)

DRONE_COLOR = (0, 180, 255)
DRONE_SIZE = 40
DRONE_SPEED = 5

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Gesture Recognition Logic ---
class GestureRecognizer:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1)
        self.results = None
        self.image_height = 0
        self.image_width = 0

    def process_frame(self, image_rgb):
        self.results = self.hands.process(image_rgb)

    def get_landmarks(self):
        if self.results and self.results.multi_hand_landmarks:
            # Return landmarks for the first detected hand
            return self.results.multi_hand_landmarks[0].landmark
        return None

    def _vector_angle(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)

    def recognize(self, landmarks):
        if not landmarks:
            return "NO_HAND", None

        # Get coordinates for key landmarks
        wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])
        thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])
        pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])

        index_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
        middle_mcp = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])
        ring_mcp = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y])
        pinky_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x, landmarks[mp_hands.HandLandmark.PINKY_MCP].y])

        # Calculate distances (relative to hand size, e.g., wrist to middle MCP)
        palm_size = np.linalg.norm(wrist - middle_mcp)
        if palm_size < 1e-6: palm_size = 1e-6 # Avoid division by zero

        dist_index = np.linalg.norm(index_tip - index_mcp) / palm_size
        dist_middle = np.linalg.norm(middle_tip - middle_mcp) / palm_size
        dist_ring = np.linalg.norm(ring_tip - ring_mcp) / palm_size
        dist_pinky = np.linalg.norm(pinky_tip - pinky_mcp) / palm_size
        dist_thumb = np.linalg.norm(thumb_tip - wrist) / palm_size # Thumb distance relative to wrist

        # Simple Rule-Based Recognition
        fingers_up = [dist_thumb > 0.8, dist_index > 1.0, dist_middle > 1.0, dist_ring > 0.9, dist_pinky > 0.9] # Thresholds might need tuning
        num_fingers_up = sum(fingers_up[1:]) # Count non-thumb fingers

        # Pointing direction (if index finger is up)
        pointing_vector = None
        if fingers_up[1]: # If index finger is up
            pip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y])
            pointing_vector = index_tip - pip
            # Normalize vector for direction only
            norm = np.linalg.norm(pointing_vector)
            if norm > 1e-6:
                pointing_vector /= norm


        if num_fingers_up == 0 and dist_thumb < 0.6:
             return "FIST", None
        elif num_fingers_up >= 4:
             return "OPEN_PALM", None
        elif num_fingers_up == 1 and fingers_up[1]: # Only index finger up
            if pointing_vector is not None:
                # Determine general direction based on vector components
                angle = math.atan2(-pointing_vector[1], pointing_vector[0]) * 180 / math.pi # Y is inverted in image coords
                if -45 < angle <= 45:
                    return "POINTING_RIGHT", pointing_vector
                elif 45 < angle <= 135:
                     return "POINTING_UP", pointing_vector # Less common for drone control this way
                elif 135 < angle or angle <= -135:
                     return "POINTING_LEFT", pointing_vector
                elif -135 < angle <= -45:
                     return "POINTING_DOWN", pointing_vector # Less common for drone control this way
            return "POINTING", pointing_vector # Fallback if direction unclear
        # Add more gestures here (e.g., thumbs up, peace sign)

        return "UNKNOWN", None

    def draw_landmarks(self, image):
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        return image

    def close(self):
        self.hands.close()

# --- Simulation Logic ---
class DroneSimulator:
    def __init__(self, screen_width, screen_height, sim_area_rect):
        self.sim_rect = sim_area_rect
        self.drone_x = self.sim_rect.centerx
        self.drone_y = self.sim_rect.centery
        self.drone_rect = pygame.Rect(0, 0, DRONE_SIZE, DRONE_SIZE)
        self.drone_rect.center = (self.drone_x, self.drone_y)
        self.target_x = self.drone_x
        self.target_y = self.drone_y
        self.speed = DRONE_SPEED
        self.command = "STOP"

    def update_command(self, gesture):
        if gesture == "FIST":
            self.command = "STOP"
        elif gesture == "OPEN_PALM":
            self.command = "UP"
        elif gesture == "POINTING_LEFT":
            self.command = "LEFT"
        elif gesture == "POINTING_RIGHT":
            self.command = "RIGHT"
        # Add more command mappings here if needed
        # elif gesture == "POINTING_DOWN":
        #     self.command = "DOWN"
        else:
            # Keep last command or stop if unknown/no hand? Let's stop.
            self.command = "STOP"


    def update_position(self):
        if self.command == "STOP":
            pass # No movement
        elif self.command == "UP":
            self.drone_y -= self.speed
        elif self.command == "DOWN":
            self.drone_y += self.speed
        elif self.command == "LEFT":
            self.drone_x -= self.speed
        elif self.command == "RIGHT":
            self.drone_x += self.speed

        # Keep drone within simulation bounds
        self.drone_x = max(self.sim_rect.left + DRONE_SIZE // 2, min(self.sim_rect.right - DRONE_SIZE // 2, self.drone_x))
        self.drone_y = max(self.sim_rect.top + DRONE_SIZE // 2, min(self.sim_rect.bottom - DRONE_SIZE // 2, self.drone_y))

        self.drone_rect.center = (self.drone_x, self.drone_y)

    def draw(self, screen):
        # Draw simulation area background
        pygame.draw.rect(screen, SIM_BG_COLOR, self.sim_rect)
        # Draw drone
        pygame.draw.rect(screen, DRONE_COLOR, self.drone_rect)
        # Draw border for sim area
        pygame.draw.rect(screen, TEXT_COLOR, self.sim_rect, 1)


# --- Main Application ---
def main():
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Kinectic Command")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    # Calculate layout rectangles
    sim_area_width = int(SCREEN_WIDTH * SIM_AREA_WIDTH_RATIO)
    video_area_width = SCREEN_WIDTH - sim_area_width
    sim_rect = pygame.Rect(0, 0, sim_area_width, SCREEN_HEIGHT)
    video_rect = pygame.Rect(sim_area_width, 0, video_area_width, SCREEN_HEIGHT)

    # Initialize Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}")
        pygame.quit()
        sys.exit()

    # Get webcam frame dimensions (needed for scaling)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        pygame.quit()
        sys.exit()
    cam_height, cam_width, _ = frame.shape

    # Calculate scaling factor to fit video into video_rect width
    video_scale_factor = video_rect.width / cam_width
    scaled_video_height = int(cam_height * video_scale_factor)
    # Center video vertically if it doesn't fill the height
    video_display_rect = pygame.Rect(video_rect.left, video_rect.top + (video_rect.height - scaled_video_height) // 2,
                                      video_rect.width, scaled_video_height)


    # Initialize Gesture Recognizer
    recognizer = GestureRecognizer()

    # Initialize Simulator
    simulator = DroneSimulator(SCREEN_WIDTH, SCREEN_HEIGHT, sim_rect)

    running = True
    last_gesture = "NONE"
    last_command_update_time = time.time()
    command_update_interval = 0.1 # Update command every 100ms

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Read Webcam Frame
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from webcam.")
            continue

        # --- Computer Vision Processing ---
        # Flip horizontally for a mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for hands
        recognizer.process_frame(image_rgb)
        landmarks = recognizer.get_landmarks()

        # Recognize Gesture
        current_time = time.time()
        gesture, _ = recognizer.recognize(landmarks) # We don't use pointing_vector directly for drone yet

        # Update simulator command based on gesture (throttled)
        if current_time - last_command_update_time > command_update_interval:
             simulator.update_command(gesture)
             last_command_update_time = current_time
             last_gesture = gesture # Store the gesture that set the command


        # Draw landmarks on the original BGR frame
        frame_with_feedback = frame.copy()
        frame_with_feedback = recognizer.draw_landmarks(frame_with_feedback)

        # Add gesture text overlay
        cv2.putText(frame_with_feedback, f"Gesture: {last_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GESTURE_TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame_with_feedback, f"Command: {simulator.command}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GESTURE_TEXT_COLOR, 2, cv2.LINE_AA)


        # --- Simulation Update ---
        simulator.update_position()

        # --- Drawing ---
        screen.fill(BG_COLOR)

        # Draw Simulation
        simulator.draw(screen)

        # Draw Video Feed Area Background
        pygame.draw.rect(screen, VIDEO_BG_COLOR, video_rect)

        # Prepare video frame for Pygame display
        # Resize frame
        frame_resized = cv2.resize(frame_with_feedback, (video_display_rect.width, video_display_rect.height))
        # Convert BGR (OpenCV) to RGB (Pygame) and rotate
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")

        # Blit video frame onto the screen
        screen.blit(frame_pygame, video_display_rect.topleft)
        pygame.draw.rect(screen, TEXT_COLOR, video_display_rect, 1) # Border for video area


        # Update Display
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    # Cleanup
    print("Exiting Kinectic Command...")
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        pygame.quit()
        cv2.destroyAllWindows() # Ensure OpenCV windows are closed on error too
        sys.exit(1)
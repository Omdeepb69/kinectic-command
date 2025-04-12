import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
from enum import Enum
from typing import Optional, List, Tuple, NamedTuple

class HandLandmark(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

class Gesture(Enum):
    UNKNOWN = "UNKNOWN"
    FIST = "FIST"
    OPEN_PALM = "OPEN_PALM"
    POINTING_UP = "POINTING_UP"
    VICTORY = "VICTORY"
    THUMBS_UP = "THUMBS_UP"

class LandmarkPoint(NamedTuple):
    x: float
    y: float
    z: float = 0.0

class HandResult(NamedTuple):
    landmarks: List[LandmarkPoint]
    world_landmarks: List[LandmarkPoint]
    handedness: str # 'Left' or 'Right'

class HandTracker:
    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

    def process_frame(self, image: np.ndarray) -> List[HandResult]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        self.results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        processed_hands = []
        if self.results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                landmarks = [LandmarkPoint(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                world_landmarks = []
                if self.results.multi_hand_world_landmarks and i < len(self.results.multi_hand_world_landmarks):
                     world_landmarks = [LandmarkPoint(lm.x, lm.y, lm.z) for lm in self.results.multi_hand_world_landmarks[i].landmark]

                handedness = "Unknown"
                if self.results.multi_handedness and i < len(self.results.multi_handedness):
                    handedness = self.results.multi_handedness[i].classification[0].label

                processed_hands.append(HandResult(landmarks=landmarks, world_landmarks=world_landmarks, handedness=handedness))
        return processed_hands

    def draw_landmarks(self, image: np.ndarray):
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return image

    def close(self):
        self.hands.close()

class GestureRecognizer:
    def __init__(self, distance_threshold_factor=0.1):
        self.distance_threshold_factor = distance_threshold_factor

    def _distance(self, p1: LandmarkPoint, p2: LandmarkPoint) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _get_finger_distances(self, landmarks: List[LandmarkPoint]) -> Tuple[float, float, float, float, float]:
        """Calculates distances from fingertips to wrist."""
        wrist = landmarks[HandLandmark.WRIST.value]
        thumb_tip = landmarks[HandLandmark.THUMB_TIP.value]
        index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP.value]
        middle_tip = landmarks[HandLandmark.MIDDLE_FINGER_TIP.value]
        ring_tip = landmarks[HandLandmark.RING_FINGER_TIP.value]
        pinky_tip = landmarks[HandLandmark.PINKY_TIP.value]

        d_thumb = self._distance(wrist, thumb_tip)
        d_index = self._distance(wrist, index_tip)
        d_middle = self._distance(wrist, middle_tip)
        d_ring = self._distance(wrist, ring_tip)
        d_pinky = self._distance(wrist, pinky_tip)

        return d_thumb, d_index, d_middle, d_ring, d_pinky

    def _get_finger_tip_pip_distances(self, landmarks: List[LandmarkPoint]) -> Tuple[float, float, float, float, float]:
        """Calculates distances between finger tips and PIP joints."""
        thumb_tip = landmarks[HandLandmark.THUMB_TIP.value]
        thumb_ip = landmarks[HandLandmark.THUMB_IP.value] # Use IP for thumb
        index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP.value]
        index_pip = landmarks[HandLandmark.INDEX_FINGER_PIP.value]
        middle_tip = landmarks[HandLandmark.MIDDLE_FINGER_TIP.value]
        middle_pip = landmarks[HandLandmark.MIDDLE_FINGER_PIP.value]
        ring_tip = landmarks[HandLandmark.RING_FINGER_TIP.value]
        ring_pip = landmarks[HandLandmark.RING_FINGER_PIP.value]
        pinky_tip = landmarks[HandLandmark.PINKY_TIP.value]
        pinky_pip = landmarks[HandLandmark.PINKY_PIP.value]

        d_thumb = self._distance(thumb_ip, thumb_tip)
        d_index = self._distance(index_pip, index_tip)
        d_middle = self._distance(middle_pip, middle_tip)
        d_ring = self._distance(ring_pip, ring_tip)
        d_pinky = self._distance(pinky_pip, pinky_tip)

        return d_thumb, d_index, d_middle, d_ring, d_pinky

    def _is_finger_extended(self, tip: LandmarkPoint, pip: LandmarkPoint, mcp: LandmarkPoint) -> bool:
        """Check if finger is relatively straight."""
        # Simple check: tip should be further from MCP than PIP is
        # More robust: check angle, or compare tip-pip distance to pip-mcp distance
        dist_tip_mcp = self._distance(tip, mcp)
        dist_pip_mcp = self._distance(pip, mcp)
        # Check if tip is significantly further away than pip
        # Also check if tip is above pip (lower y-coordinate in image space)
        return dist_tip_mcp > dist_pip_mcp * 1.2 and tip.y < pip.y

    def _is_thumb_extended(self, tip: LandmarkPoint, ip: LandmarkPoint, mcp: LandmarkPoint) -> bool:
        """Check if thumb is extended."""
        dist_tip_mcp = self._distance(tip, mcp)
        dist_ip_mcp = self._distance(ip, mcp)
        # Check if tip is significantly further away than ip
        return dist_tip_mcp > dist_ip_mcp * 1.2

    def recognize(self, hand_result: HandResult) -> Gesture:
        landmarks = hand_result.landmarks
        if not landmarks:
            return Gesture.UNKNOWN

        # Finger extension status based on tip vs PIP/MCP positions
        thumb_extended = self._is_thumb_extended(
            landmarks[HandLandmark.THUMB_TIP.value],
            landmarks[HandLandmark.THUMB_IP.value],
            landmarks[HandLandmark.THUMB_MCP.value]
        )
        index_extended = self._is_finger_extended(
            landmarks[HandLandmark.INDEX_FINGER_TIP.value],
            landmarks[HandLandmark.INDEX_FINGER_PIP.value],
            landmarks[HandLandmark.INDEX_FINGER_MCP.value]
        )
        middle_extended = self._is_finger_extended(
            landmarks[HandLandmark.MIDDLE_FINGER_TIP.value],
            landmarks[HandLandmark.MIDDLE_FINGER_PIP.value],
            landmarks[HandLandmark.MIDDLE_FINGER_MCP.value]
        )
        ring_extended = self._is_finger_extended(
            landmarks[HandLandmark.RING_FINGER_TIP.value],
            landmarks[HandLandmark.RING_FINGER_PIP.value],
            landmarks[HandLandmark.RING_FINGER_MCP.value]
        )
        pinky_extended = self._is_finger_extended(
            landmarks[HandLandmark.PINKY_TIP.value],
            landmarks[HandLandmark.PINKY_PIP.value],
            landmarks[HandLandmark.PINKY_MCP.value]
        )

        extended_fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
        num_extended = sum(extended_fingers)

        # Gesture Rules
        if num_extended == 0 and not thumb_extended:
             # Check if fingertips are close to palm (e.g., wrist or center)
             palm_center_y = (landmarks[HandLandmark.WRIST.value].y + landmarks[HandLandmark.MIDDLE_FINGER_MCP.value].y) / 2
             tips_below_pip = all(landmarks[tip.value].y > landmarks[pip.value].y for tip, pip in [
                 (HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP),
                 (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
                 (HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP),
                 (HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP)
             ])
             if tips_below_pip:
                 return Gesture.FIST

        if num_extended == 4 and thumb_extended: # Allow some flexibility for thumb
            return Gesture.OPEN_PALM

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Check if index finger tip is significantly higher than others
            if landmarks[HandLandmark.INDEX_FINGER_TIP.value].y < landmarks[HandLandmark.INDEX_FINGER_PIP.value].y:
                 return Gesture.POINTING_UP

        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.VICTORY

        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
             # Check thumb tip is above thumb MCP
             if landmarks[HandLandmark.THUMB_TIP.value].y < landmarks[HandLandmark.THUMB_MCP.value].y:
                 return Gesture.THUMBS_UP

        return Gesture.UNKNOWN


class SimulationObject:
    def __init__(self, x, y, width, height, color=(0, 128, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.target_pos = pygame.Vector2(x, y)
        self.speed = 5.0

    def set_target(self, x: Optional[float] = None, y: Optional[float] = None):
        if x is not None:
            self.target_pos.x = x
        if y is not None:
            self.target_pos.y = y

    def move(self, dx: float = 0, dy: float = 0):
         self.target_pos.x += dx
         self.target_pos.y += dy

    def update(self, dt: float):
        current_pos = pygame.Vector2(self.rect.center)
        direction = self.target_pos - current_pos

        if direction.length_squared() > 1: # Avoid jitter when close
            direction.normalize_ip()
            move_vec = direction * self.speed #* dt * 60 # Adjust speed based on dt if needed
            new_pos = current_pos + move_vec

            # Clamp position within screen bounds (adjust as needed)
            # new_pos.x = max(self.rect.width / 2, min(screen_width - self.rect.width / 2, new_pos.x))
            # new_pos.y = max(self.rect.height / 2, min(screen_height - self.rect.height / 2, new_pos.y))

            self.rect.center = new_pos


    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.color, self.rect)
        # Draw target position marker (optional)
        # pygame.draw.circle(surface, (255, 0, 0), (int(self.target_pos.x), int(self.target_pos.y)), 5)


class KinecticCommandApp:
    def __init__(self, webcam_id=0, sim_width=600, sim_height=480):
        self.webcam_id = webcam_id
        self.sim_width = sim_width
        self.sim_height = sim_height

        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam {self.webcam_id}")

        # Get actual frame dimensions from webcam
        ret, frame = self.cap.read()
        if not ret:
             raise IOError(f"Cannot read frame from webcam {self.webcam_id}")
        self.cam_height, self.cam_width = frame.shape[:2]


        self.hand_tracker = HandTracker(max_hands=1)
        self.gesture_recognizer = GestureRecognizer()

        pygame.init()
        pygame.display.set_caption("Kinectic Command Simulation")
        self.screen = pygame.display.set_mode((self.sim_width, self.sim_height))
        self.clock = pygame.time.Clock()
        self.sim_object = SimulationObject(sim_width // 2, sim_height // 2, 50, 50)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.gesture_text = ""
        self.last_command = Gesture.UNKNOWN

    def _map_gesture_to_command(self, gesture: Gesture, hand_result: Optional[HandResult]):
        command_changed = (gesture != self.last_command)
        self.last_command = gesture

        if gesture == Gesture.FIST:
            # Example: Move down
            self.sim_object.move(dy=10)
        elif gesture == Gesture.OPEN_PALM:
            # Example: Stop / Hover at current target
             pass # No change in target
        elif gesture == Gesture.POINTING_UP:
            # Example: Move up
            self.sim_object.move(dy=-10)
        elif gesture == Gesture.THUMBS_UP:
             # Example: Move right
             self.sim_object.move(dx=10)
        elif gesture == Gesture.VICTORY:
             # Example: Move left
             self.sim_object.move(dx=-10)
        # Add more mappings here

        # Example: Control position directly with index finger tip
        # if gesture == Gesture.POINTING_UP and hand_result:
        #     index_tip = hand_result.landmarks[HandLandmark.INDEX_FINGER_TIP.value]
        #     # Map normalized camera coords to simulation coords
        #     target_x = index_tip.x * self.sim_width
        #     target_y = index_tip.y * self.sim_height
        #     self.sim_object.set_target(target_x, target_y)

        # Clamp target position to screen bounds
        self.sim_object.target_pos.x = max(self.sim_object.rect.width / 2, min(self.sim_width - self.sim_object.rect.width / 2, self.sim_object.target_pos.x))
        self.sim_object.target_pos.y = max(self.sim_object.rect.height / 2, min(self.sim_height - self.sim_object.rect.height / 2, self.sim_object.target_pos.y))


    def run(self):
        running = True
        while running and self.cap.isOpened():
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Read frame from webcam
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Process frame for hand tracking
            hand_results = self.hand_tracker.process_frame(frame)

            current_gesture = Gesture.UNKNOWN
            active_hand_result = None
            if hand_results:
                active_hand_result = hand_results[0] # Assuming max_hands=1
                # Recognize gesture
                current_gesture = self.gesture_recognizer.recognize(active_hand_result)
                self.gesture_text = f"Gesture: {current_gesture.value}"

                # Draw landmarks on the frame
                frame = self.hand_tracker.draw_landmarks(frame)
            else:
                self.gesture_text = "Gesture: No Hand Detected"
                self.last_command = Gesture.UNKNOWN # Reset command if hand lost

            # Map gesture to simulation command
            self._map_gesture_to_command(current_gesture, active_hand_result)

            # Update simulation state
            dt = self.clock.tick(60) / 1000.0 # Delta time in seconds
            self.sim_object.update(dt)

            # --- Drawing ---
            # Draw simulation
            self.screen.fill((30, 30, 30)) # Dark background
            self.sim_object.draw(self.screen)
            pygame.display.flip()

            # Draw gesture text on webcam frame
            cv2.putText(frame, self.gesture_text, (10, 30), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the annotated webcam frame
            cv2.imshow('Kinectic Command - Hand Tracking', frame)

            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
                running = False

        # Cleanup
        self.hand_tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == '__main__':
    try:
        app = KinecticCommandApp(webcam_id=0)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up resources if initialization failed partially
        if 'app' in locals() and hasattr(app, 'cap') and app.cap.isOpened():
            app.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
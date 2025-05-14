import cv2
import numpy as np
import pygame
import mediapipe as mp
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
from OpenGL.GLUT import glutSolidSphere, glutSolidCube

class IronManDroneARSystem:
    # Gesture types
    PEACE = "PEACE"            # ✌️ - Switch to normal mode
    THREE_FINGERS = "THREE"    # ☝️✌️ - Switch to rotation mode
    ROCK_ON = "ROCK_ON"        # 🤘 - Switch to height mode
    THUMBS_UP = "THUMBS_UP"    # 👍 - Increase speed
    THUMBS_DOWN = "THUMBS_DOWN"  # 👎 - Decrease speed
    FIST = "FIST"              # ✊ - Stop
    OPEN_PALM = "OPEN_PALM"    # ✋ - Hover
    POINTING = "POINTING"      # ☝️ - Move drone in pointing direction
    PINCH = "PINCH"            # 👌 - Follow mode

    # Control modes
    MODE_NORMAL = "NORMAL"
    MODE_ROTATION = "ROTATION"
    MODE_HEIGHT = "HEIGHT"
    
    # Colors
    IRON_MAN_ORANGE = (255, 165, 0)
    IRON_MAN_BLUE = (0, 191, 255)
    WHITE = (255, 255, 255)
    
    def __init__(self, width=1280, height=720):
        # Initialize window dimensions
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Iron Man Drone AR Control System")
        
        # Set up OpenGL perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.aspect_ratio, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Initialize drone properties
        self.drone_position = [0, 0, -5]  # x, y, z
        self.drone_rotation = [0, 0, 0]   # pitch, yaw, roll
        self.drone_target_position = [0, 0, -5]
        self.drone_target_rotation = [0, 0, 0]
        self.drone_size = 0.5
        self.drone_speed = 0.1
        self.vertical_speed = 0.05
        self.rotation_speed = 2.0
        
        # Initialize hand tracking variables
        self.hand_landmarks = None
        self.hand_position = [0, 0, 0]
        self.index_finger_tip = [0, 0, 0]
        self.thumb_tip = [0, 0, 0]
        self.is_grabbing = False
        self.last_gesture = None
        self.current_gesture = None
        
        # Control state
        self.current_mode = self.MODE_NORMAL
        self.following_hand = False
        self.is_flying = False
        self.battery_level = 100
        self.signal_strength = 95
        self.flight_timer = 0
        self.start_time = time.time()
        
        # Drone rotor animation
        self.rotor_angle = 0
        self.rotor_speed = 10
        
        # Fonts for HUD
        self.font = pygame.font.SysFont('Arial', 20)
        self.large_font = pygame.font.SysFont('Arial', 32)
        
        # Create texture for background
        self.texture_id = glGenTextures(1)
    
    def load_background_texture(self, frame):
        """Load the camera frame as an OpenGL texture for background"""
        frame = cv2.flip(frame, 0)  # Flip for OpenGL coordinates
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
    
    def draw_background(self):
        """Draw the camera frame as background"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_drone(self):
        """Draw the 3D drone model"""
        glPushMatrix()
        
        # Position and rotate the drone
        glTranslatef(self.drone_position[0], self.drone_position[1], self.drone_position[2])
        glRotatef(self.drone_rotation[0], 1, 0, 0)  # Pitch
        glRotatef(self.drone_rotation[1], 0, 1, 0)  # Yaw
        glRotatef(self.drone_rotation[2], 0, 0, 1)  # Roll
        
        # Body material (Iron Man red/gold)
        body_material = [0.8, 0.2, 0.0, 1.0]  # Red-orange
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, body_material)
        
        # Draw drone body (central sphere)
        glPushMatrix()
        glScalef(0.8, 0.3, 1.0)  # Flatten into drone shape
        glutSolidSphere(self.drone_size * 0.5, 16, 16)
        glPopMatrix()
        
        # Draw drone arms
        arm_material = [0.1, 0.1, 0.1, 1.0]  # Dark gray
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, arm_material)
        
        # Arm length and thickness
        arm_length = self.drone_size * 1.2
        arm_thickness = self.drone_size * 0.08
        
        # Draw 4 arms
        for angle in range(0, 360, 90):
            glPushMatrix()
            glRotatef(angle, 0, 1, 0)
            
            # Draw arm
            glPushMatrix()
            glTranslatef(arm_length/2, 0, 0)
            glScalef(arm_length, arm_thickness, arm_thickness)
            glutSolidCube(1.0)
            glPopMatrix()
            
            # Draw motor housing at end of arm
            motor_material = [0.7, 0.5, 0.0, 1.0]  # Gold
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, motor_material)
            
            glPushMatrix()
            glTranslatef(arm_length, 0, 0)
            glutSolidSphere(self.drone_size * 0.15, 8, 8)
            glPopMatrix()
            
            # Draw spinning rotor
            rotor_material = [0.3, 0.3, 0.3, 0.7]  # Semi-transparent gray
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, rotor_material)
            
            glPushMatrix()
            glTranslatef(arm_length, 0.05, 0)
            glRotatef(self.rotor_angle, 0, 1, 0)
            
            # Draw rotor blades
            glBegin(GL_TRIANGLES)
            # Blade 1
            glVertex3f(0, 0, 0)
            glVertex3f(self.drone_size * 0.4, 0, self.drone_size * 0.1)
            glVertex3f(self.drone_size * 0.4, 0, -self.drone_size * 0.1)
            # Blade 2
            glVertex3f(0, 0, 0)
            glVertex3f(-self.drone_size * 0.4, 0, self.drone_size * 0.1)
            glVertex3f(-self.drone_size * 0.4, 0, -self.drone_size * 0.1)
            glEnd()
            
            glPopMatrix()
            
            glPopMatrix()
        
        # Draw status LED
        if self.is_flying:
            led_material = [0.0, 1.0, 0.0, 1.0]  # Green when flying
        else:
            led_material = [1.0, 0.0, 0.0, 1.0]  # Red when stopped
        
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, led_material)
        glPushMatrix()
        glTranslatef(0, self.drone_size * 0.3, 0)
        glutSolidSphere(self.drone_size * 0.08, 8, 8)
        glPopMatrix()
        
        glPopMatrix()
    
    def draw_hud(self):
        """Draw Iron Man style HUD elements"""
        # Switch to 2D orthographic projection for HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Helper function to draw text
        def draw_text(text, position, color, font=self.font):
            text_surface = font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            text_width, text_height = text_surface.get_size()
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glRasterPos2d(position[0], position[1])
            glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glDisable(GL_BLEND)
        
        # Draw title and status
        draw_text("IRON MAN DRONE CONTROL SYSTEM", (20, self.height - 40), self.IRON_MAN_ORANGE, self.large_font)
        draw_text(f"MODE: {self.current_mode}", (20, self.height - 80), self.IRON_MAN_BLUE)
        
        # Draw drone stats
        draw_text(f"Position: X:{self.drone_position[0]:.2f} Y:{self.drone_position[1]:.2f} Z:{self.drone_position[2]:.2f}", 
                 (20, 60), self.WHITE)
        draw_text(f"Rotation: P:{self.drone_rotation[0]:.1f} Y:{self.drone_rotation[1]:.1f} R:{self.drone_rotation[2]:.1f}", 
                 (20, 30), self.WHITE)
        
        # Status indicators on right side
        draw_text(f"Battery: {self.battery_level}%", (self.width - 200, self.height - 40), self.WHITE)
        draw_text(f"Signal: {self.signal_strength}%", (self.width - 200, self.height - 70), self.WHITE)
        
        # Flight timer
        minutes, seconds = divmod(int(self.flight_timer), 60)
        draw_text(f"Flight Time: {minutes:02d}:{seconds:02d}", (self.width - 200, self.height - 100), self.WHITE)
        
        # Speed indicators
        draw_text(f"Speed: {self.drone_speed:.2f}", (self.width - 200, 60), self.WHITE)
        draw_text(f"Vert Speed: {self.vertical_speed:.2f}", (self.width - 200, 30), self.WHITE)
        
        # Current gesture
        if self.current_gesture:
            draw_text(f"Gesture: {self.current_gesture}", (self.width // 2 - 100, 30), self.IRON_MAN_ORANGE)
        
        # Draw circular HUD elements (Iron Man style)
        def draw_circle(center, radius, color, thickness=2, filled=False):
            num_segments = 36
            glLineWidth(thickness)
            glColor3f(color[0]/255, color[1]/255, color[2]/255)
            
            if filled:
                glBegin(GL_POLYGON)
            else:
                glBegin(GL_LINE_LOOP)
                
            for i in range(num_segments):
                theta = 2.0 * math.pi * i / num_segments
                x = radius * math.cos(theta)
                y = radius * math.sin(theta)
                glVertex2f(center[0] + x, center[1] + y)
            glEnd()
        
        # Draw HUD circles
        draw_circle((self.width // 2, self.height // 2), 100, self.IRON_MAN_BLUE, 2)
        draw_circle((self.width // 2, self.height // 2), 103, self.IRON_MAN_ORANGE, 1)
        draw_circle((self.width // 2, self.height // 2), 150, self.IRON_MAN_BLUE, 1)
        
        # Artificial horizon
        horizon_center = (self.width // 2, self.height // 2)
        horizon_width = 300
        pitch_offset = -self.drone_rotation[0] * 1.5  # Scale pitch for visibility
        
        # Draw horizon line
        glLineWidth(2)
        glColor3f(self.IRON_MAN_BLUE[0]/255, self.IRON_MAN_BLUE[1]/255, self.IRON_MAN_BLUE[2]/255)
        glBegin(GL_LINES)
        glVertex2f(horizon_center[0] - horizon_width/2, horizon_center[1] + pitch_offset)
        glVertex2f(horizon_center[0] + horizon_width/2, horizon_center[1] + pitch_offset)
        glEnd()
        
        # Reset OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def detect_hand_gestures(self, frame):
        """Process the frame to detect hand landmarks and recognize gestures"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Reset hand landmarks
        self.hand_landmarks = None
        
        if results.multi_hand_landmarks:
            # Get the first detected hand
            self.hand_landmarks = results.multi_hand_landmarks[0]
            hand_landmarks = self.hand_landmarks
            
            # Extract key points
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Store 3D positions of key points
            self.hand_position = [
                (wrist.x - 0.5) * self.width / 250,  # Scale for OpenGL coordinates
                -(wrist.y - 0.5) * self.height / 250,
                wrist.z * 10
            ]
            
            self.index_finger_tip = [
                (index_tip.x - 0.5) * self.width / 250,
                -(index_tip.y - 0.5) * self.height / 250,
                index_tip.z * 10
            ]
            
            self.thumb_tip = [
                (thumb_tip.x - 0.5) * self.width / 250,
                -(thumb_tip.y - 0.5) * self.height / 250,
                thumb_tip.z * 10
            ]
            
            # Calculate distances for gesture recognition
            thumb_index_distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2 + 
                (thumb_tip.z - index_tip.z)**2
            )
            
            # Check if fingers are extended or not
            index_extended = index_tip.y < index_pip.y
            middle_extended = middle_tip.y < middle_pip.y
            ring_extended = ring_tip.y < ring_pip.y
            pinky_extended = pinky_tip.y < pinky_pip.y
            thumb_extended = thumb_tip.x > thumb_ip.x if wrist.x < 0.5 else thumb_tip.x < thumb_ip.x
            
            # Recognize gestures
            self.last_gesture = self.current_gesture
            
            # PEACE sign (index and middle extended, others closed)
            if index_extended and middle_extended and not ring_extended and not pinky_extended:
                self.current_gesture = self.PEACE
            
            # THREE FINGERS (index, middle, and ring extended)
            elif index_extended and middle_extended and ring_extended and not pinky_extended:
                self.current_gesture = self.THREE_FINGERS
            
            # ROCK ON (index and pinky extended, others closed)
            elif index_extended and not middle_extended and not ring_extended and pinky_extended:
                self.current_gesture = self.ROCK_ON
            
            # THUMBS UP (only thumb extended upward)
            elif thumb_extended and thumb_tip.y < wrist.y and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                self.current_gesture = self.THUMBS_UP
            
            # THUMBS DOWN (only thumb extended downward)
            elif thumb_extended and thumb_tip.y > wrist.y and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                self.current_gesture = self.THUMBS_DOWN
            
            # FIST (no fingers extended)
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
                self.current_gesture = self.FIST
            
            # OPEN PALM (all fingers extended)
            elif index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
                self.current_gesture = self.OPEN_PALM
            
            # POINTING (only index extended)
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                self.current_gesture = self.POINTING
            
            # PINCH (thumb and index close together)
            elif thumb_index_distance < 0.05:
                self.current_gesture = self.PINCH
                self.is_grabbing = True
            else:
                self.is_grabbing = False
            
            # Draw hand landmarks on the frame (optional for debugging)
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            return True
        else:
            self.current_gesture = None
            return False
    
    def update_command(self):
        """Update drone commands based on detected gestures"""
        if self.current_gesture == self.PEACE:
            # Switch to normal mode
            self.current_mode = self.MODE_NORMAL
        
        elif self.current_gesture == self.THREE_FINGERS:
            # Switch to rotation mode
            self.current_mode = self.MODE_ROTATION
        
        elif self.current_gesture == self.ROCK_ON:
            # Switch to height mode
            self.current_mode = self.MODE_HEIGHT
        
        elif self.current_gesture == self.THUMBS_UP:
            # Increase speed
            self.drone_speed = min(0.5, self.drone_speed + 0.05)
            self.vertical_speed = min(0.25, self.vertical_speed + 0.025)
        
        elif self.current_gesture == self.THUMBS_DOWN:
            # Decrease speed
            self.drone_speed = max(0.05, self.drone_speed - 0.05)
            self.vertical_speed = max(0.025, self.vertical_speed - 0.025)
        
        elif self.current_gesture == self.FIST:
            # Stop the drone
            self.is_flying = False
            self.following_hand = False
            self.drone_target_position = self.drone_position.copy()
        
        elif self.current_gesture == self.OPEN_PALM:
            # Hover in place
            self.is_flying = True
            self.following_hand = False
        
        elif self.current_gesture == self.PINCH:
            # Enable follow mode
            self.is_flying = True
            self.following_hand = True
        
        elif self.current_gesture == self.POINTING:
            # Move in pointing direction
            self.is_flying = True
            self.following_hand = False
            
            if self.current_mode == self.MODE_NORMAL:
                # Normal mode - horizontal movement
                move_x = self.index_finger_tip[0] - self.hand_position[0]
                move_z = self.index_finger_tip[2] - self.hand_position[2]
                
                # Scale movement by distance from wrist to fingertip
                magnitude = np.sqrt(move_x**2 + move_z**2)
                
                if magnitude > 0.1:  # Threshold to avoid small movements
                    self.drone_target_position[0] += move_x * self.drone_speed
                    self.drone_target_position[2] += move_z * self.drone_speed
            
            elif self.current_mode == self.MODE_ROTATION:
                # Rotation mode - change yaw based on horizontal finger position
                yaw_change = (self.index_finger_tip[0] - self.hand_position[0]) * self.rotation_speed
                self.drone_target_rotation[1] += yaw_change
                
                # Change pitch based on vertical finger position
                pitch_change = (self.index_finger_tip[1] - self.hand_position[1]) * self.rotation_speed
                self.drone_target_rotation[0] = max(-30, min(30, self.drone_target_rotation[0] + pitch_change))
            
            elif self.current_mode == self.MODE_HEIGHT:
                # Height mode - change altitude based on vertical finger position
                height_change = (self.index_finger_tip[1] - self.hand_position[1])
                
                if abs(height_change) > 0.1:  # Threshold to avoid small movements
                    self.drone_target_position[1] += height_change * self.vertical_speed
        
        # Follow hand if in follow mode
        if self.following_hand and self.hand_landmarks is not None:
            self.drone_target_position = [
                self.hand_position[0],
                self.hand_position[1],
                self.hand_position[2] - 2  # Keep drone in front of hand
            ]
        
        # Constrain drone position within bounds
        self.drone_target_position[0] = max(-5, min(5, self.drone_target_position[0]))
        self.drone_target_position[1] = max(-3, min(3, self.drone_target_position[1]))
        self.drone_target_position[2] = max(-10, min(-2, self.drone_target_position[2]))
    
    def update_drone_physics(self):
        """Update drone position and rotation with smooth transitions"""
        # Update position with easing
        for i in range(3):
            self.drone_position[i] += (self.drone_target_position[i] - self.drone_position[i]) * 0.1
        
        # Update rotation with easing
        for i in range(3):
            # Find shortest path for rotation (handle 360 degree wrap)
            diff = (self.drone_target_rotation[i] - self.drone_rotation[i]) % 360
            if diff > 180:
                diff -= 360
            
            self.drone_rotation[i] += diff * 0.1
        
        # Update rotor animation
        base_rotor_speed = 10
        if self.is_flying:
            # Increase rotor speed based on movement and vertical speed
            movement_factor = sum(abs(self.drone_position[i] - self.drone_target_position[i]) for i in range(3))
            vert_factor = abs(self.drone_position[1] - self.drone_target_position[1]) * 10
            
            self.rotor_speed = base_rotor_speed + movement_factor * 20 + vert_factor * 30
        else:
            # Slow down rotors when not flying
            self.rotor_speed = max(0, self.rotor_speed - 1)
        
        self.rotor_angle = (self.rotor_angle + self.rotor_speed) % 360
        
        # Update battery simulation
        if self.is_flying:
            self.battery_level = max(0, self.battery_level - 0.01)
            
            # Signal strength variation
            self.signal_strength = min(100, max(70, self.signal_strength + np.random.uniform(-0.5, 0.5)))
        
        # Update flight timer
        if self.is_flying:
            self.flight_timer = time.time() - self.start_time
    
    def run(self):
        """Main loop for the AR drone control system"""
        running = True
        clock = pygame.time.Clock()
        
        print("Iron Man Drone AR Control System")
        print("Gestures:")
        print(f"  {self.PEACE} (✌️) - Switch to normal mode")
        print(f"  {self.THREE_FINGERS} (☝️✌️) - Switch to rotation mode")
        print(f"  {self.ROCK_ON} (🤘) - Switch to height mode")
        print(f"  {self.THUMBS_UP} (👍) - Increase speed")
        print(f"  {self.THUMBS_DOWN} (👎) - Decrease speed")
        print(f"  {self.FIST} (✊) - Stop")
        print(f"  {self.OPEN_PALM} (✋) - Hover")
        print(f"  {self.POINTING} (☝️) - Move drone in pointing direction")
        print(f"  {self.PINCH} (👌) - Follow mode")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
            
            # Capture and process frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Mirror the frame horizontally for more intuitive control
            frame = cv2.flip(frame, 1)
            
            # Detect hand gestures
            hand_detected = self.detect_hand_gestures(frame)
            
            # Update drone commands based on gestures
            if hand_detected:
                self.update_command()
            
            # Update drone physics
            self.update_drone_physics()
            
            # Load camera frame as background texture
            self.load_background_texture(frame)
            
            # Clear the screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Draw the background (camera frame)
            self.draw_background()
            
            # Set up the 3D view
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Draw the drone
            self.draw_drone()
            
            # Draw HUD elements
            self.draw_hud()
            
            # Update the display
            pygame.display.flip()
            
            # Cap at 30 FPS
            clock.tick(30)
        
        # Clean up
        self.cap.release()
        self.hands.close()
        pygame.quit()

def main():
    try:
        from OpenGL.GLUT import glutInit
        # Initialize GLUT for 3D primitives
        glutInit()
    except:
        print("GLUT initialization failed. Make sure PyOpenGL is properly installed.")
        return
    
    try:
        # Create and run the AR drone control system
        drone_system = IronManDroneARSystem()
        drone_system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
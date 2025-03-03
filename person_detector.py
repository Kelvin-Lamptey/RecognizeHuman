import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import serial.tools.list_ports


class PersonDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Initialize MediaPipe Pose Detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Movement tracking variables
        self.last_poses = []
        self.movement_threshold = 15  # REDUCED from 30 to be more sensitive to small movements
        self.sleeping_counter = 0
        self.awake_counter = 0
        self.state_threshold = 10  # Number of consistent frames to change state
        self.current_state = "No person detected"

    def calculate_movement(self, landmarks):
        """Calculate movement between current and previous poses"""
        if not self.last_poses:
            self.last_poses.append(landmarks)
            return 100  # Default to high movement on first detection
        
        # Compare with the last pose
        last_pose = self.last_poses[-1]
        
        # Calculate distance between key points (nose, shoulders, eyes)
        movement = 0
        key_points = [0, 1, 2, 3, 4, 5, 6]  # Indices for key face/upper body points
        
        for point in key_points:
            if point < len(landmarks) and point < len(last_pose):
                current = landmarks[point]
                previous = last_pose[point]
                
                # Calculate Euclidean distance
                distance = np.sqrt((current.x - previous.x)**2 + 
                                  (current.y - previous.y)**2)
                movement += distance
        
        # Keep last 5 poses for tracking
        self.last_poses.append(landmarks)
        if len(self.last_poses) > 5:
            self.last_poses.pop(0)
            
        return movement * 1000  # Scale up for easier threshold setting

    def is_horizontal(self, landmarks):
        """Check if the person might be in a horizontal (sleeping) position"""
        # Get left and right shoulder positions
        if not landmarks:
            return False
            
        # Check if shoulders and hips indicate horizontal posture
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate the angle of the torso relative to vertical
        shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_avg_x = (left_hip.x + right_hip.x) / 2
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate angle (0 is vertical, 90 is horizontal)
        dx = hip_avg_x - shoulder_avg_x
        dy = hip_avg_y - shoulder_avg_y
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        # If angle is close to horizontal (90 degrees)
        return angle > 60  # INCREASED from 45 to 60 degrees to avoid misclassifying seated positions

    def detect_person_state(self, frame):
        """Detect if a person is present and determine if they're awake or sleeping"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image for face detection
        face_results = self.face_detection.process(rgb_frame)
        face_detected = face_results.detections is not None and len(face_results.detections) > 0
        
        # Process the image for pose detection
        pose_results = self.pose.process(rgb_frame)
        pose_detected = pose_results.pose_landmarks is not None
        
        # If no person detected
        if not face_detected and not pose_detected:
            self.sleeping_counter = 0
            self.awake_counter = 0
            self.current_state = "No person detected"
            return frame, self.current_state
        
        # Draw face detections
        if face_detected:
            for detection in face_results.detections:
                self.mp_drawing.draw_detection(frame, detection)
        
        # Process pose if detected
        if pose_detected:
            # Draw pose landmarks on the image
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Extract landmarks for analysis
            landmarks = []
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.append(landmark)
            
            # Calculate movement
            movement = self.calculate_movement(landmarks)
            
            # Check if horizontal
            horizontal_posture = self.is_horizontal(landmarks)
            
            # Determine if sleeping or awake
            if movement < self.movement_threshold and horizontal_posture:
                self.sleeping_counter += 1
                self.awake_counter = 0
            else:
                self.awake_counter += 1
                self.sleeping_counter = 0
            
            # Update state based on counters
            if self.sleeping_counter > self.state_threshold:
                self.current_state = "Person is sleeping"
            elif self.awake_counter > self.state_threshold:
                self.current_state = "Person is awake"
            
            # Display movement value for debugging
            cv2.putText(frame, f"Movement: {movement:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, self.current_state

def setup_serial_connection():
    """Set up and return a serial connection to Arduino"""
    ports = serial.tools.list_ports.comports()
    ports_list = []
    
    print("Available ports:")
    for i, port in enumerate(ports):
        ports_list.append(str(port))
        print(f"[{i}] {port}")
    
    try:
        if not ports_list:
            print("No serial ports found. Arduino communication disabled.")
            return None
            
        # Ask for port selection with COM3 as default
        selection = input("Select a port number (or press Enter for default): ")
        if selection.strip() == "":
            com = "3"  # Default to COM3 if no selection
            print(f"Using default COM3")
        else:
            com = selection
        
        # Find the selected port
        selected_port = None
        for port in ports_list:
            if port.startswith(f"COM{com}"):
                selected_port = f"COM{com}"
                break
        
        if not selected_port:
            print(f"COM{com} not found. Arduino communication disabled.")
            return None
            
        # Set up and open the serial connection
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = selected_port
        ser.timeout = 1
        ser.open()
        
        print(f"Connected to Arduino on {selected_port}")
        return ser
        
    except Exception as e:
        print(f"Error setting up serial connection: {e}")
        return None

def send_to_arduino(arduino, state):
    """Send the appropriate command to Arduino based on state"""
    if not arduino or not arduino.is_open:
        return False
        
    try:
        if state == "No person detected" or state == "Person is sleeping":
            arduino.write("OFF".encode('utf-8'))  # Added newline terminator
            return "Sent OFF signal to Arduino"
        else:
            arduino.write("ON".encode('utf-8'))  # Added newline terminator
            return "Sent ON signal to Arduino"
    except Exception as e:
        print(f"Error sending to Arduino: {e}")
        return f"Error: {e}"

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = PersonDetector()
    
    # Set up Arduino communication once before the loop
    arduino = setup_serial_connection()
    
    last_print_time = time.time()
    last_arduino_command = ""
    
    # Add state tracking variables
    current_state = "No person detected"
    state_change_time = time.time()
    last_sent_command = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Process frame
        processed_frame, state = detector.detect_person_state(frame)
        
        # Check if state has changed
        if state != current_state:
            current_state = state
            state_change_time = time.time()
        
        # Display state on frame
        cv2.putText(processed_frame, state, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update Arduino every 2 seconds
        current_time = time.time()
        if current_time - last_print_time >= 2:
            print(f"Current state: {state}")
            
            # Determine command to send based on current state and timing
            command_to_send = None
            if state == "Person is awake":
                # Send ON immediately when person is awake
                command_to_send = "ON"
            elif (state == "No person detected" or state == "Person is sleeping"):
                # Check if 5 seconds have elapsed before sending OFF
                time_in_state = current_time - state_change_time
                remaining_time = max(0, 5 - time_in_state)
                
                if time_in_state >= 5:
                    command_to_send = "OFF"
                    print(f"Sending OFF after waiting 5 seconds")
                else:
                    print(f"Waiting {remaining_time:.1f} more seconds before sending OFF")
            
            # Send command if determined
            if command_to_send and command_to_send != last_sent_command:
                try:
                    if command_to_send == "OFF":
                        status = send_to_arduino(arduino, "No person detected")
                    else:
                        status = send_to_arduino(arduino, "Person is awake")
                    
                    print(status)
                    last_sent_command = command_to_send
                except Exception as e:
                    print(f"Error sending command: {e}")
                
            last_print_time = current_time
        
        # Display the frame
        cv2.imshow('Person Detection', processed_frame)
        
        # Quit on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Clean up
    if arduino and arduino.is_open:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
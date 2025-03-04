import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import serial.tools.list_ports
import torch


class PersonDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            enable_segmentation=False,
            static_image_mode=False
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Movement tracking variables
        self.last_poses = []
        self.movement_threshold = 15
        self.sleeping_counter = 0
        self.awake_counter = 0
        self.state_threshold = 10
        self.current_state = "No person detected"

    def calculate_movement(self, landmarks):
        """Calculate movement between current and previous poses"""
        if not self.last_poses:
            self.last_poses.append(landmarks)
            return 100
        
        last_pose = self.last_poses[-1]
        movement = 0
        key_points = [0, 1, 2, 3, 4, 5, 6]  # Key points for movement detection
        
        for point in key_points:
            if point < len(landmarks) and point < len(last_pose):
                current = landmarks[point]
                previous = last_pose[point]
                distance = np.sqrt((current.x - previous.x)**2 + (current.y - previous.y)**2)
                movement += distance
        
        self.last_poses.append(landmarks)
        if len(self.last_poses) > 5:
            self.last_poses.pop(0)
            
        return movement * 1000

    def is_horizontal(self, landmarks):
        """Check if the person might be in a horizontal (sleeping) position"""
        if not landmarks:
            return False
            
        # Check if shoulders and hips indicate horizontal posture
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate angle of torso relative to vertical
        shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_avg_x = (left_hip.x + right_hip.x) / 2
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate angle (0 is vertical, 90 is horizontal)
        dx = hip_avg_x - shoulder_avg_x
        dy = hip_avg_y - shoulder_avg_y
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        return angle > 60  # Threshold for horizontal position

    def detect_person_state(self, frame):
        """Detect person and determine if they're awake or sleeping"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        state = "No person detected"
        
        if results.pose_landmarks:
            # Calculate movement and check posture
            landmarks = list(results.pose_landmarks.landmark)
            movement = self.calculate_movement(landmarks)
            horizontal_posture = self.is_horizontal(landmarks)
            
            # Update state counters
            if movement < self.movement_threshold and horizontal_posture:
                self.sleeping_counter += 1
                self.awake_counter = 0
            else:
                self.awake_counter += 1
                self.sleeping_counter = 0
            
            # Determine state
            if self.sleeping_counter > self.state_threshold:
                state = "Person is sleeping"
                color = (128, 0, 128)  # Purple for sleeping
            elif self.awake_counter > self.state_threshold:
                state = "Person is awake"
                color = (0, 255, 0)  # Green for awake
            else:
                state = "Detecting..."
                color = (255, 255, 0)  # Yellow for uncertain
            
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
            )
        
        # Add state text to frame
        cv2.putText(frame, state, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, state

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
import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import serial.tools.list_ports


class PersonDetector:
    def __init__(self):
        # Initialize MediaPipe Pose with multi-person detection
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
        
        # Movement tracking variables - per person
        self.last_poses = {}
        self.movement_threshold = 15
        self.sleeping_counters = {}
        self.awake_counters = {}
        self.state_threshold = 10
        self.current_state = "No person detected"
        
        # Person tracking
        self.tracked_poses = {}  # Store previous poses for tracking

    def calculate_movement(self, landmarks, person_id):
        """Calculate movement between current and previous poses for a specific person"""
        if person_id not in self.last_poses:
            self.last_poses[person_id] = []
        
        if not self.last_poses[person_id]:
            self.last_poses[person_id].append(landmarks)
            return 100
        
        last_pose = self.last_poses[person_id][-1]
        movement = 0
        key_points = [0, 1, 2, 3, 4, 5, 6]
        
        for point in key_points:
            if point < len(landmarks) and point < len(last_pose):
                current = landmarks[point]
                previous = last_pose[point]
                distance = np.sqrt((current.x - previous.x)**2 + (current.y - previous.y)**2)
                movement += distance
        
        self.last_poses[person_id].append(landmarks)
        if len(self.last_poses[person_id]) > 5:
            self.last_poses[person_id].pop(0)
            
        return movement * 1000

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
        """Detect multiple people and determine if they're awake or sleeping"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Track states for all detected people
        person_states = []
        active_people = set()
        
        # Process multiple poses if detected
        if results.pose_landmarks:
            # Get current pose center
            current_pose = results.pose_landmarks
            nose = current_pose.landmark[0]
            center_x = int(nose.x * frame_width)
            center_y = int(nose.y * frame_height)
            current_center = (center_x, center_y)
            
            # Try to match with existing tracked poses
            matched_id = None
            min_distance = float('inf')
            
            for pid, (prev_center, _) in self.tracked_poses.items():
                distance = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                if distance < min_distance and distance < 100:  # 100 pixel threshold
                    min_distance = distance
                    matched_id = pid
            
            # If no match found, create new ID
            if matched_id is None:
                matched_id = len(self.tracked_poses)
            
            # Update tracked poses
            self.tracked_poses[matched_id] = (current_center, current_pose)
            
            # Convert landmarks to list
            landmarks = []
            for landmark in current_pose.landmark:
                landmarks.append(landmark)
            
            # Calculate movement and posture
            movement = self.calculate_movement(landmarks, matched_id)
            horizontal_posture = self.is_horizontal(landmarks)
            
            # Initialize counters if needed
            if matched_id not in self.sleeping_counters:
                self.sleeping_counters[matched_id] = 0
            if matched_id not in self.awake_counters:
                self.awake_counters[matched_id] = 0
            
            # Determine state for this person
            if movement < self.movement_threshold and horizontal_posture:
                self.sleeping_counters[matched_id] += 1
                self.awake_counters[matched_id] = 0
            else:
                self.awake_counters[matched_id] += 1
                self.sleeping_counters[matched_id] = 0
            
            # Get person state
            person_state = "Unknown"
            if self.sleeping_counters[matched_id] > self.state_threshold:
                person_state = "sleeping"
            elif self.awake_counters[matched_id] > self.state_threshold:
                person_state = "awake"
            
            person_states.append(person_state)
            active_people.add(matched_id)
            
            # Draw landmarks with different colors for each person
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            base_color = colors[matched_id % len(colors)]
            color = base_color if person_state == "awake" else (128, 0, 128)
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, current_pose, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
            )
            
            # Add state label above person
            text_position = (center_x - 50, center_y - 30)
            cv2.putText(frame, f"Person {matched_id+1}: {person_state}", 
                      text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Clean up tracking for people no longer detected
        tracked_people = set(self.sleeping_counters.keys())
        for old_id in tracked_people - active_people:
            del self.sleeping_counters[old_id]
            del self.awake_counters[old_id]
            if old_id in self.last_poses:
                del self.last_poses[old_id]
            if old_id in self.tracked_poses:
                del self.tracked_poses[old_id]
        
        # Determine overall state with priority to awake people
        if not person_states:
            overall_state = "No person detected"
        elif "awake" in person_states:
            overall_state = "Person is awake"
        elif "sleeping" in person_states:
            overall_state = "Person is sleeping"
        else:
            overall_state = "No person detected"
        
        # Add overall state count to frame
        awake_count = person_states.count("awake")
        sleeping_count = person_states.count("sleeping")
        cv2.putText(frame, f"Awake: {awake_count}, Sleeping: {sleeping_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, overall_state

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
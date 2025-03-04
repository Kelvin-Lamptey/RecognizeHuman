import cv2
import mediapipe as mp
import numpy as np
import torch
import time

class MultiPersonDetector:
    def __init__(self):
        # Initialize YOLO
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.classes = [0]  # Only detect persons
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Movement tracking variables - per person
        self.last_poses = {}
        self.movement_threshold = 15
        self.sleeping_counters = {}
        self.awake_counters = {}
        self.state_threshold = 10
        
        # Person tracking
        self.tracked_poses = {}

    def detect_persons_yolo(self, frame):
        """Detect persons using YOLO and return their bounding boxes"""
        results = self.yolo_model(frame)
        persons = []
        
        for detection in results.xyxy[0]:  # xyxy format
            if detection[5] == 0:  # class 0 is person
                x1, y1, x2, y2, conf = detection[:5]
                if conf > 0.5:  # Confidence threshold
                    persons.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf)
                    })
        
        return persons

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
        if not landmarks:
            return False
            
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_avg_x = (left_hip.x + right_hip.x) / 2
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        
        dx = hip_avg_x - shoulder_avg_x
        dy = hip_avg_y - shoulder_avg_y
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        return angle > 60

    def detect_person_state(self, frame):
        """Detect multiple people and determine if they're awake or sleeping"""
        # Detect persons using YOLO
        persons = self.detect_persons_yolo(frame)
        person_states = []
        active_people = set()
        
        # Process each detected person
        for idx, person in enumerate(persons):
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add padding to bounding box
            padding = 20
            height, width = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Crop person from frame
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_frame.size == 0:
                continue
            
            # Process the cropped frame
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            ) as pose:
                rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if not results.pose_landmarks:
                    continue
                
                # Adjust landmark coordinates to original frame
                adjusted_landmarks = results.pose_landmarks
                for landmark in adjusted_landmarks.landmark:
                    landmark.x = (landmark.x * (x2 - x1) + x1) / width
                    landmark.y = (landmark.y * (y2 - y1) + y1) / height
                
                # Calculate movement and check posture
                landmarks = list(adjusted_landmarks.landmark)
                movement = self.calculate_movement(landmarks, idx)
                horizontal_posture = self.is_horizontal(landmarks)
                
                # Initialize counters if needed
                if idx not in self.sleeping_counters:
                    self.sleeping_counters[idx] = 0
                if idx not in self.awake_counters:
                    self.awake_counters[idx] = 0
                
                # Update state counters
                if movement < self.movement_threshold and horizontal_posture:
                    self.sleeping_counters[idx] += 1
                    self.awake_counters[idx] = 0
                else:
                    self.awake_counters[idx] += 1
                    self.sleeping_counters[idx] = 0
                
                # Determine person state
                state = "unknown"
                if self.sleeping_counters[idx] > self.state_threshold:
                    state = "sleeping"
                elif self.awake_counters[idx] > self.state_threshold:
                    state = "awake"
                
                person_states.append(state)
                active_people.add(idx)
                
                # Draw pose landmarks with different colors based on state and person ID
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                base_color = colors[idx % len(colors)]
                color = base_color if state == "awake" else (128, 0, 128)
                
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    frame, adjusted_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
                )
                
                # Add state label above person
                cv2.putText(frame, f"Person {idx+1}: {state}", 
                          (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Clean up tracking for people no longer detected
        tracked_people = set(self.sleeping_counters.keys())
        for old_id in tracked_people - active_people:
            del self.sleeping_counters[old_id]
            del self.awake_counters[old_id]
            if old_id in self.last_poses:
                del self.last_poses[old_id]
            if old_id in self.tracked_poses:
                del self.tracked_poses[old_id]
        
        # Determine overall state
        if not person_states:
            overall_state = "No person detected"
        elif "awake" in person_states:
            overall_state = "Person is awake"
        elif "sleeping" in person_states:
            overall_state = "Person is sleeping"
        else:
            overall_state = "No person detected"
        
        # Add counts to frame
        awake_count = person_states.count("awake")
        sleeping_count = person_states.count("sleeping")
        cv2.putText(frame, f"Awake: {awake_count}, Sleeping: {sleeping_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, overall_state

def main():
    cap = cv2.VideoCapture(0)
    detector = MultiPersonDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Process frame
        processed_frame, state = detector.detect_person_state(frame)
        
        # Display the frame
        cv2.imshow('Multi-Person Detection', processed_frame)
        
        # Quit on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
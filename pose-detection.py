import cv2
import mediapipe as mp
import os
import numpy as np
import csv

# Initialize MediaPipe Pose solution
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Function to process and annotate frames or images
def annotate_frame(frame, pose):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get pose landmarks
    results = pose.process(frame_rgb)

    angles = [None, None, None, None]
    # Annotate the frame if landmarks are detected
    if results.pose_landmarks:
        # Calculate and display joint angles
        # Get coordinates
        landmarks = results.pose_landmarks.landmark
        # Left arm
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Right arm
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Left leg
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Right leg
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
        # Calculate angles
        l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_leg_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Visualize angle
        joints = {
            "Left Arm": (l_elbow, l_arm_angle),
            "Right Arm": (r_elbow, r_arm_angle),
            "Left Leg": (l_knee, l_leg_angle),
            "Right Leg": (r_knee, r_leg_angle)
        }
        angles = [l_arm_angle, r_arm_angle, l_leg_angle, r_leg_angle]
        
        for i, (joint, (_, angle)) in enumerate(joints.items()):
            text_position = (width - 200, 30 + i * 30)  # fixed position near top-right
            cv2.putText(frame, f"{joint}: {int(angle)}°", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
  
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return frame, angles

# Function to calculate joint angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Function to handle video
def process_files(file_list, is_image=True):
    with mp_pose.Pose(
            static_image_mode=is_image,  # Static mode for images, dynamic for videos
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        for file in file_list:
            # Process a video
            cap = cv2.VideoCapture(file)
            if not cap.isOpened():
                print(f"Error opening video {file}")
                continue

            # Initialize CSV file
            angle_csv = f'joint_angles_{os.path.basename(file)}.csv'
            with open(angle_csv, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Frame', 'Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle'])
                frame_index = 0

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("End of video or error reading frame.")
                        break

                    # Process the current frame
                    annotated_frame, angles = annotate_frame(frame, pose)

                    writer.writerow([frame_index] + list(map(float, angles)))
                    frame_index += 1

                    """
                    # Uncomment to display the annotated frame
                    # Display the annotated frame
                    window_name = f'Pose Detection - Video {os.path.basename(file)}'
                    cv2.imshow(window_name, annotated_frame)
                            
                    # Exit on pressing 'q'
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        print("Exiting video display...")
                        break
                    """
            cap.release()
            print(f"Joint angles saved as {angle_csv}")
        cv2.destroyAllWindows()


file_list = input("Enter video file paths separated by commas: ").split(',')
file_list = [file.strip() for file in file_list]
process_files(file_list, is_image=False)


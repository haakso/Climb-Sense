import cv2
import mediapipe as mp
import os
import numpy as np
import csv


"""

Adding comments to test how creating a branch in vs code

"""

# Initialize MediaPipe Pose solution
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def compute_com(joints):
    return np.mean(joints, axis =0)

# Function to process and annotate frames or images
def annotate_frame(frame, pose, prev_data):
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
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]

        # Right arm
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

        # Left leg
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]

        # Right leg
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                
        # Calculate angles
        l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_leg_angle = calculate_angle(r_hip, r_knee, r_ankle)

        
        com = compute_com([l_shoulder, r_shoulder, l_hip, r_hip])
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()

        speed = 0

        if prev_data["com"] is not None and prev_data["time"] is not None:
            dt = timestamp - prev_data["time"]
            if dt > 0:
                velocity = (com - prev_data["com"]) / dt
                speed = np.linalg.norm(velocity)
                cv2.putText(frame, f"CoM Speed: {speed:.2f} m/s", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        prev_data["com"] = com
        prev_data["time"] = timestamp


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
    return angles, speed

# Function to calculate joint angle in R^3

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
    return np.degrees(angle)

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

            # Ask the user if the move was successful for this video
            success_label = input(f"Was the move successful in '{file}'? (Yes/No): ").strip().capitalize()
            if success_label not in ['Yes', 'No']:
                print("Invalid input. Defaulting to 'No'.")
                success_label = 'No'


            # Initialize CSV file
            angle_csv = f'joint_angles_{os.path.basename(file)}.csv'
            with open(angle_csv, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Frame', 'Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle', 'CoM Velocity', 'Move Successful'])
                frame_index = 0

                prev_data = {"com": None, "time": None}

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("End of video or error reading frame.")
                        break

                    # Process the current frame
                    angles, speed = annotate_frame(frame, pose, prev_data)

                    writer.writerow([frame_index] + np.array(angles).flatten().tolist() + [speed])
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


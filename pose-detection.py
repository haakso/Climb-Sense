import cv2
import mediapipe as mp
import os

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

    # Annotate the frame if landmarks are detected
    if results.pose_landmarks:
        # Print coordinates of the nose (example)
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height})'
        )

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return frame

# Function to handle multiple inputs for photo and video
def process_files(file_list, is_image=True):
    with mp_pose.Pose(
            static_image_mode=is_image,  # Static mode for images, dynamic for videos
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        for file in file_list:
            if is_image:
                # Process an image
                img = cv2.imread(file)
                if img is None:
                    print(f"Error loading image {file}")
                else:
                    annotated_img = annotate_frame(img, pose)
                    # Save and show the annotated image
                    output_img_path = f'annotated_{os.path.basename(file)}'
                    cv2.imwrite(output_img_path, annotated_img)
                    print(f"Annotated image saved as {output_img_path}")
                    cv2.imshow(f'Pose Detection - Image: {os.path.basename(file)}', annotated_img)
                    cv2.waitKey(0)  # Wait for any key press to close image window
                    cv2.destroyAllWindows()
            else:
                # Process a video
                cap = cv2.VideoCapture(file)
                if not cap.isOpened():
                    print(f"Error opening video {file}")
                    continue

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Define the codec and create VideoWriter object
                output_video = f'annotated_{os.path.basename(file)}'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("End of video or error reading frame.")
                        break

                    # Process the current frame
                    annotated_frame = annotate_frame(frame, pose)

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                    # Display the annotated frame
                    window_name = f'Pose Detection - Video {os.path.basename(file)}'
                    cv2.imshow(window_name, annotated_frame)

                    # Exit on pressing 'q'
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        print("Exiting video display...")
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()
                print(f"Annotated video saved as {output_video}")

# Ask user to choose between "photo", "video", or "webcam"
print("Choose mode of detection:")
print("1. Photo")
print("2. Video")
print("3. Webcam")
choice = input("Enter choice (1/2/3): ")

if choice == '1':
    # Photo Mode: Process multiple image files
    file_list = input("Enter image file paths separated by commas: ").split(',')
    file_list = [file.strip() for file in file_list]
    process_files(file_list, is_image=True)

elif choice == '2':
    # Video Mode: Process multiple video files
    file_list = input("Enter video file paths separated by commas: ").split(',')
    file_list = [file.strip() for file in file_list]
    process_files(file_list, is_image=False)

elif choice == '3':
    # Webcam Mode: Process live webcam feed
    cap = cv2.VideoCapture(0)  # Open the webcam (default is 0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        output_video = 'annotated_webcam_output.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                # Capture frame-by-frame from the webcam
                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture frame.")
                    break

                # Process the current frame
                annotated_frame = annotate_frame(frame, pose)

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # Display the frame with annotations
                cv2.imshow('Live Pose Detection - Webcam', annotated_frame)

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting webcam feed...")
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Annotated webcam video saved as {output_video}")

else:
    print("Invalid choice. Please restart the program and select 1, 2, or 3.")

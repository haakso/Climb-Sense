# pose-detection-using-mediapipe-photo-video-real-time-
This project leverages Google's MediaPipe library to perform pose detection on images, videos, and live webcam feeds. The code processes these inputs, detects key body landmarks, and annotates them with visual markers.

Features
Image Processing: Annotates multiple images with pose detection landmarks.
Video Processing: Annotates multiple videos frame-by-frame with pose detection landmarks.
Live Webcam Processing: Detects and annotates poses in real-time from a webcam feed.
Requirements
Python 3.x
OpenCV
MediaPipe
Installing Dependencies
You can install the required dependencies using pip. To set up the project, follow these steps:

Clone the repository:

bash
Copy code

git clone https://github.com/AVSAkash/pose-detection.git

cd pose-detection

Install the required Python packages:

bash
Copy code

pip install opencv-python mediapipe

Usage
Step 1: Choose Mode of Detection
You will be prompted to choose one of the following modes:

Photo: Process multiple image files.
Video: Process multiple video files.
Webcam: Process live webcam feed.

Step 2: Provide Input Files
In Photo and Video modes, provide file paths to images or videos (comma-separated) for processing.
In Webcam mode, the program will automatically detect and annotate poses in real-time.

Step 3: Results
The program will annotate the detected body landmarks (e.g., nose, shoulders, elbows) and save the results as new images or videos.
The processed output will be saved with a filename prefixed with annotated_.
In Webcam mode, the live feed will also be saved as an annotated video.
Example Commands

Photo Mode:

Input: image1.jpg, image2.jpg
Output: Annotated images annotated_image1.jpg, annotated_image2.jpg.

Video Mode:

Input: video1.mp4, video2.mp4
Output: Annotated videos annotated_video1.avi, annotated_video2.avi.

Webcam Mode:
To stop the Webcam mode press "q" in ur keyboard.

Live webcam feed will be processed and saved as annotated_webcam_output.avi.

Code Explanation

The program utilizes MediaPipe's Pose model to detect and annotate body landmarks in real-time.
OpenCV is used to handle image/video processing and display the annotated results.
The pose landmarks are drawn on the image/video using predefined drawing styles from MediaPipe.

Example Output

Image Output: Annotated image showing pose landmarks.
Video Output: Annotated video with keypoints overlaid on the person in each frame.
Webcam Output: Real-time pose annotation with body landmarks displayed on webcam feed.

Troubleshooting

Ensure that the input files (images or videos) are in the correct format and accessible.
If you encounter any issues with webcam access, ensure that the device drivers are installed and the webcam is functioning properly.                                                                                                                                                                           

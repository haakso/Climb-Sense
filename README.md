# ClimbSense: An Analysis of Moonboard Climbing using Computer Vision

We will use the intel realsense d415 to capture video footage of climbers performing individual climbing movements. This project leverages Google's MediaPipe library to perform pose detection of the videos captured by the d415. MediaPipe's Pose model detects and annotates body landmarks and OpenCV handles the video processing and displays the annotated results.
Pose landmarks are drawn on the video using predefined drawing styles from MediaPipe. The program outputs an annotated video with key points overlaid on the person in each frame.


References:
https://github.com/AVSAkash/pose-detection.git
https://github.com/nicknochnack/MediaPipePoseEstimation


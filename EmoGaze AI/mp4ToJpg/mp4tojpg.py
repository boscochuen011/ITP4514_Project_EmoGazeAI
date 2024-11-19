import cv2
import os

# Open the video file
cap = cv2.VideoCapture('input.mp4')

# Ensure the output directory exists
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip to get 10 frames per minute
skip_frames = int(fps * 60 / 10)

frame_count = 0
saved_count = 0
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Only save every skip_frames-th frame
    if frame_count % skip_frames == 0:
        # Write the frame to an image file
        output_path = os.path.join(output_dir, f'Look_set2Train_{saved_count}.jpg')
        cv2.imwrite(output_path, frame)
        saved_count += 1

    frame_count += 1

# Release the VideoCapture
cap.release()
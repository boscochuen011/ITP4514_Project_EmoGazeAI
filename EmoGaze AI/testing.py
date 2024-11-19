import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import librosa
import matplotlib.pyplot as plt
import warnings

# Load the detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the gaze and emotion models
gaze_model = load_model("gaze_model.keras")
emotion_model = load_model("emotion_model.keras")

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Load images from the `emotion_training_set` directory
train_generator = datagen.flow_from_directory(
    'dataset/emotion_set/emotion_training_set',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Get class indices
class_indices = train_generator.class_indices

# Create a dictionary that maps from indices to labels
indices_to_labels = {v: k for k, v in class_indices.items()}

# Open the video file
cap = cv2.VideoCapture('input.mp4')

# Get the video's frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (frame_width, frame_height))

import numpy as np
import cv2

def extract_eye(image, start, end):
    shape = predictor(image, rect)  # Assuming `shape` is the shape predictor's output for a face
    points = shape.parts()[start:end]  # Get the points that form the eye
    points = np.array([(p.x, p.y) for p in points])  # Convert to numpy array
    x, y, w, h = cv2.boundingRect(points)  # Create a bounding box around the eye
    eye = image[y:y+h, x:x+w]  # Crop the image to the bounding box
    return eye

gaze_probabilities = []

while cap.isOpened():
    # Read a frame from the video
    ret, img = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector(gray)

    # For each detected face
    for rect in faces:
        # Get the landmarks
        shape = predictor(gray, rect)

        # Draw the face landmarks on the image
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            
        # Extract the eyes
        left_eye = extract_eye(gray, 36, 42)
        right_eye = extract_eye(gray, 42, 48)

        # Resize the eye images to the expected input size
        left_eye = cv2.resize(left_eye, (64, 64))
        right_eye = cv2.resize(right_eye, (64, 64))

        # Normalize pixel values to [0, 1]
        left_eye = left_eye.astype('float32') / 255.0
        right_eye = right_eye.astype('float32') / 255.0

        # Add a color channel
        left_eye = np.expand_dims(left_eye, axis=-1)
        right_eye = np.expand_dims(right_eye, axis=-1)

        # Add a batch dimension
        left_eye = np.expand_dims(left_eye, axis=0)
        right_eye = np.expand_dims(right_eye, axis=0)

        # Predict the gaze
        gaze_prediction_left = gaze_model.predict(left_eye)
        gaze_prediction_right = gaze_model.predict(right_eye)

        # Preprocess the face image for the gaze and emotion models

        # Preprocess the face image for the gaze model
        face_gray = cv2.cvtColor(img[rect.top():rect.bottom(), rect.left():rect.right()], cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (64, 64))
        face_gray = face_gray.astype("float") / 255.0
        face_gray = np.expand_dims(face_gray, axis=-1)  # Add an extra dimension for channels
        face_gray = img_to_array(face_gray)
        face_gray = np.expand_dims(face_gray, axis=0)  # Add an extra dimension for batch

        # Predict the gaze
        gaze_prediction = gaze_model.predict(face_gray)

        # Convert the gray image to RGB for the emotion model
        face_rgb = cv2.cvtColor(img[rect.top():rect.bottom(), rect.left():rect.right()], cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (64, 64))
        face_rgb = img_to_array(face_rgb)
        face_rgb = np.expand_dims(face_rgb, axis=0)
        face_rgb = preprocess_input(face_rgb)

        # Predict with the emotion model
        emotion_prediction = emotion_model.predict(face_gray)
        
        gaze_prediction = gaze_model.predict(face_gray)
        gaze_confidence = gaze_prediction[0][0]
        gaze_probabilities.append(gaze_confidence)

        # Get the index of the predicted emotion
        emotion_index = np.argmax(emotion_prediction)

        # Get the label of the predicted emotion
        emotion_label = indices_to_labels[emotion_index]

        # Get the confidence of the predicted emotion
        emotion_confidence = np.max(emotion_prediction)

        # Predict the gaze
        gaze_prediction = gaze_model.predict(face_gray)

        # Get the gaze status and confidence
        gaze_confidence = gaze_prediction[0][0]
        gaze_status = "Looking at the camera" if gaze_confidence > 0.5 else "Not looking at the camera"
        
        # Convert confidence to percentage
        gaze_confidence_percent = gaze_confidence * 100

       # Define the box color
        green_intensity = int(gaze_confidence * 255)
        red_intensity = int((1 - gaze_confidence) * 255)
        box_color = (0, red_intensity, green_intensity)

        # Draw the rectangle around the face
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), box_color, 2)

       # Assume gaze_confidence is a value between 0 and 1, where 1 is looking at the camera
        gaze_confidence = gaze_confidence_percent / 100.0

        # Interpolate between red (0,0,255) and green (0,255,0) based on gaze_confidence
        box_color = (0, int(255 * gaze_confidence), int(255 * (1 - gaze_confidence)))

        # Draw the rectangle around the face
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), box_color, 2)

        text = f"{emotion_label} ({emotion_confidence*100:.2f}%), {gaze_status} ({gaze_confidence_percent:.2f}%)"

        # Define the text color
        color = (255, 255, 255)

        # Define the font scale and thickness
        font_scale = 0.5
        font_thickness = 1

        # Get the size of the text box
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Define the text box's top-left and bottom-right points
        text_box_tl = (rect.left(), rect.top() - text_height - 10)
        text_box_br = (max(rect.right(), rect.left() + text_width), rect.top())

        # Draw a rectangle to make the text more visible
        cv2.rectangle(img, text_box_tl, text_box_br, box_color, cv2.FILLED)

        # Display the predicted emotion, confidence, and gaze status above the rectangle
        cv2.putText(img, text, (rect.left(), rect.top()-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Write the processed frame to the output video
        out.write(img)

# Release the VideoCapture and VideoWriter and close the video files
# Ensure the result directory exists
result_dir = 'output'
os.makedirs(result_dir, exist_ok=True)

# Load the audio file
y, sr = librosa.load('input.mp4')

# Calculate the volume for each frame
S, phase = librosa.magphase(librosa.stft(y))
volumes = librosa.amplitude_to_db(S, ref=np.max)

# Create a new figure
plt.figure()

# Plot the average volume over time
plt.plot(np.mean(volumes, axis=0))

# Add title and labels for the axes
plt.title('Volume Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Volume (dB)')

# Save the figure before showing it
plt.savefig(os.path.join(result_dir, 'Volume_check.png'))

warnings.filterwarnings("ignore")

# Create a new figure for the gaze probabilities plot
plt.figure()

# Create a range for the x-axis
x = range(len(gaze_probabilities))

# Create the plot
plt.plot(x, gaze_probabilities)

# Add a horizontal line at y=0.5
plt.axhline(y=0.5, color='r', linestyle='--')

# Add title and labels
plt.title('Gaze probabilities over time')
plt.xlabel('Frame')
plt.ylabel('Gaze probability')

# Save the figure before showing it
plt.savefig(os.path.join(result_dir, 'gaze_probabilities.png'))

cap.release()
out.release()

cv2.destroyAllWindows()
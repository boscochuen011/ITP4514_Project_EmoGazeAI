import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator

# Define paths
base_dir = 'dataset/emotion_set'
train_dir = os.path.join(base_dir, 'emotion_training_set')
val_dir = os.path.join(base_dir, 'emotion_test_set')

# Define image dimensions and batch size
img_width, img_height = 64, 64
batch_size = 64

# Prepare training data with real-time data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Define model architecture
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.5))

# Get the number of classes
num_classes = len(train_generator.class_indices)

emotion_model.add(Dense(num_classes, activation='softmax'))

# Define initial learning rate
initial_learning_rate = 0.0001

# Define decay steps
decay_steps = 10000

# Define decay rate
decay_rate = 1e-6

# Define learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate, staircase=True)

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate=initial_learning_rate), 
                      metrics=['accuracy'])
# Calculate steps_per_epoch in a way that it is never 0
steps_per_epoch = max(1, train_generator.samples // batch_size)
validation_steps = max(1, validation_generator.samples // batch_size)

# Train the model with more epochs
history = emotion_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=200) # Increase number of epochs

# Save the model
emotion_model.save('emotion_model.keras')

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

result_dir = 'training_result'
os.makedirs(result_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'emotion_training_history.png'))
plt.show()
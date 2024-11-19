import os
import cv2
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
base_dir = 'dataset/gaze_set'
train_dir = os.path.join(base_dir, 'gaze_training_set')
val_dir = os.path.join(base_dir, 'gaze_test_set')

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
    class_mode='binary') # Change to binary

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='binary') # Change to binary
# Define model architecture
gaze_model = Sequential()

gaze_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
gaze_model.add(MaxPooling2D(pool_size=(2, 2)))
gaze_model.add(Dropout(0.3))

gaze_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gaze_model.add(MaxPooling2D(pool_size=(2, 2)))
gaze_model.add(Dropout(0.3))

gaze_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
gaze_model.add(MaxPooling2D(pool_size=(2, 2)))
gaze_model.add(Dropout(0.4))

gaze_model.add(Flatten())
gaze_model.add(Dense(2048, activation='relu'))
gaze_model.add(Dropout(0.5))

gaze_model.add(Dense(1, activation='sigmoid')) 

# Define learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = LegacyAdam(learning_rate=lr_schedule)

# Compile the model
gaze_model.compile(loss='binary_crossentropy',
                   optimizer=optimizer, 
                   metrics=['accuracy'])

# Define steps
steps_per_epoch = math.ceil(train_generator.samples / batch_size)
validation_steps = math.ceil(validation_generator.samples / batch_size)

checkpoint = ModelCheckpoint('gaze_model.h5', save_best_only=True)

# Train the model
history = gaze_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=100, 
    callbacks=[checkpoint])

# Save the model
gaze_model.save('gaze_model.keras')

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
plt.savefig(os.path.join(result_dir, 'gaze_training_history.png'))
plt.show()
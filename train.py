# Author @ Nydia R. Varela-Rosales, M. Engel
# Version v1 2024
# Description : train model based on FFT from simulation snapshots of the cqc phase at different epsilon and T values
# Requires    : tensorflow, keras, time, data FFT images

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import time

# Parameters
testName     = "v3_" 
metrics_file = testName+'training_metrics.txt' # name of model version 
batch_size   = 16                              # ideal batch size for running on GPU at 16 cores 
sizeImage    = 512                             # image size
channels     = 1                               # image channels, no colored images (channels=1)
dropout      = 0.2                             # dropout in hidden layers
waitTime     = 40                              # wait time before finishing training
epochsNumber = 100                             # max. number of epochs for training
learningRate = 1e-4                            # learning rate for optimization

class SaveMetricsCallback(keras.callbacks.Callback):
    def __init__(self, filepath, interval=60):  # periodicity of the output of the target metric
        super(SaveMetricsCallback, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.last_save_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        if current_time - self.last_save_time >= self.interval:
            with open(self.filepath, 'a') as file:
                # Print epoch #, loss, accuracy, val. loss in metrics_file
                file.write(f"Epoch {epoch+1}: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}, Validation Loss = {logs['val_loss']}\n")
                self.last_save_time = current_time


# Define augmentation
train_datagen      = ImageDataGenerator(rescale=1./255, rotation_range=0, zoom_range=0.,
                                   width_shift_range=0., height_shift_range=0., shear_range=0.0, horizontal_flip=False)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(sizeImage, sizeImage),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

# Building a more complex CNN model with additional layers and dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(sizeImage, sizeImage, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(dropout),  
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with a possibly lower learning rate
model.compile(optimizer=Adam(learning_rate=learningRate), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks to save the best model so far
early_stopping = EarlyStopping(monitor='val_loss', patience=waitTime, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(testName+'best_chirality_detector_early.keras', save_best_only=True, monitor='val_loss')

# Include your custom callback in your training
callbacks = [
    early_stopping,
    model_checkpoint,
    SaveMetricsCallback(filepath=metrics_file)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochsNumber,
    callbacks=callbacks
)

# Save the model
model.save(testName+'test_chirality_detector_early.keras')
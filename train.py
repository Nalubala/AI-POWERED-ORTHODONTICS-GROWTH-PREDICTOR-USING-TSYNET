import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    "image_size": (256, 256),
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001,
    "num_classes": 3,  # Update based on your dataset
    "dataset_path": r"C:\Users\NALUBALA ARJUN\Dropbox\PC\Desktop\Orthodontic1\RawImage\TrainingData",  # Update with your dataset path
    "model_save_path": "models/tsynet_ortho_predictor.keras"  # Changed from .h5 to .keras
}

# Ensure model save directory exists
os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)

# Create TSYNET model architecture
def create_tsynet_model(input_shape, num_classes):
    model = Sequential([
        # Feature extraction blocks
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # TSYNET-specific blocks
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Prediction blocks
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Prepare data generators
def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator

# Train the model
def train_model():
    # Create model
    input_shape = CONFIG["image_size"] + (3,)
    model = create_tsynet_model(input_shape, CONFIG["num_classes"])
    
    # Prepare data
    train_generator, val_generator = prepare_data()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CONFIG["model_save_path"],
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        train_generator,
        epochs=CONFIG["epochs"],
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    print("Starting TSYNET Orthodontic Growth Predictor Training...")
    model = train_model()
    print(f"Training complete. Model saved to {CONFIG['model_save_path']}")
    
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

print(tf.test.is_built_with_cuda())  # Should return True


# Load images from directory function remains the same
def load_images_from_directory(base_dir):
    data = []
    labels = []
    shapes = ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'circle', 'star']
    
    for shape in shapes:
        shape_dir = os.path.join(base_dir, shape)
        for filename in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(shape)
    
    return np.array(data), np.array(labels)


# Path to your dataset
data_dir = 'C:/Users/rohit/Downloads/symmetry-master/detection-master/sortedSet'

# Load and preprocess the images and labels
data, labels = load_images_from_directory(data_dir)

def preprocess(data, labels):
    data = data.reshape((data.shape[0], 128, 128, 1)).astype('float32') / 255.0  # Normalize the data
    label_encoder = LabelEncoder()  # Encode the labels
    labels = to_categorical(label_encoder.fit_transform(labels))  # One-hot encode the labels
    return data, labels, label_encoder

x, y, label_encoder = preprocess(data, labels)

# Stratified split to ensure each class is equally represented
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1, stratify=y)

# Data augmentation for training
datagen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)
datagen_train.fit(x_train)

# Build the enhanced CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile with AdamW optimizer
model.compile(optimizer=AdamW(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model with the enhanced setup
history = model.fit(datagen_train.flow(x_train, y_train, batch_size=32),
                    epochs=5,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print(f"Loss on test data: {loss:.4f}")

# Save the trained model
model.save('../models/identifier.h5')

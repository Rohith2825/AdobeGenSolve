# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO

# Set up constants
DATASET_DIR = '/content/standard_object_shape-2'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VAL_DIR = os.path.join(DATASET_DIR, 'valid')

# Define class mapping
CLASS_MAPPING = {
    '0': 'circle',
    '1': 'cross',
    '2': 'heptagon',
    '3': 'hexagon',
    '4': 'octagon',
    '5': 'pentagon',
    '6': 'quarter circle',
    '7': 'rectangle',
    '8': 'semi circle',
    '9': 'square',
    '10': 'star',
    '11': 'trapezoid',
    '12': 'triangle'
}

# Function to explore dataset
def explore_dataset(dataset_dir, dataset_name):
    num_images = len(os.listdir(os.path.join(dataset_dir, 'images')))
    num_labels = len(os.listdir(os.path.join(dataset_dir, 'labels')))
    
    print(f"Exploring {dataset_name} dataset:")
    print(f"Number of images: {num_images}")
    print(f"Number of labels (annotations): {num_labels}")

    class_counts = {}
    for label_file in os.listdir(os.path.join(dataset_dir, 'labels')):
        with open(os.path.join(dataset_dir, 'labels', label_file), 'r') as lf:
            for line in lf:
                class_name = line.strip().split()[0]
                class_label = CLASS_MAPPING.get(class_name, class_name)
                if class_label in class_counts:
                    class_counts[class_label] += 1
                else:
                    class_counts[class_label] = 1

    classes, counts = zip(*class_counts.items())
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.title(f'Class Distribution in {dataset_name} Dataset')
    plt.xlabel('Classes (Shapes)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Function to display random sample images
def display_random_samples(image_dir, num_samples=5):
    sample_images = np.random.choice(os.listdir(image_dir), num_samples, replace=False)
    plt.figure(figsize=(15, 5))
    for i, image_filename in enumerate(sample_images):
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Sample Image {i+1}')
    plt.show()

# Function to train YOLOv8 model
def train_yolov8_model(config_path, epochs=3):
    model = YOLO("yolov8n.pt")
    model.train(data=config_path, epochs=epochs)
    return model

# Function to perform inference with YOLOv8
def perform_inference(model, image_dir, output_dir):
    results = model.predict(source=image_dir, save=True, save_txt=True)
    return results

# Function to download dataset from Roboflow
def download_dataset(api_key, workspace, project_name, version, save_format):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download(save_format)
    return dataset



# Main script execution
if __name__ == "__main__":
    # Step 1: Download and explore dataset
    api_key = "Q3aYoueljqQ1S5nhgaYY"
    workspace = "hku-uas-deprecated-sobt2"
    project_name = "standard_object_shape"
    version = 2
    save_format = "yolov8"

    download_dataset(api_key, workspace, project_name, version, save_format)



    rf = Roboflow(api_key="Q3aYoueljqQ1S5nhgaYY")
    project = rf.workspace("hku-uas-deprecated-sobt2").project("standard_object_shape")
    dataset = project.version(2).download("yolov8")

    explore_dataset(TRAIN_DIR, 'Training')
    explore_dataset(TEST_DIR, 'Testing')
    explore_dataset(VAL_DIR, 'Validation')

    display_random_samples(TRAIN_DIR)

    # Step 2: Train YOLOv8 model
    config_path = os.path.join(DATASET_DIR, 'data.yaml')
    model = train_yolov8_model(config_path, epochs=3)

    # Step 3: Perform inference on test images
    output_dir = os.path.join(DATASET_DIR, 'results')
    perform_inference(model, TEST_DIR, output_dir)

    print("Shape detection completed successfully.")

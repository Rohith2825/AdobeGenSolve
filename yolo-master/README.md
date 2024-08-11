# YOLOv8 Shape Detection

Welcome to the YOLOv8 Shape Detection project! This repository, developed by team SSNCE for Adobe GenSolve, implements a shape detection model using YOLOv8, tailored for detecting various shapes.

## Why YOLOv8 Wasn't Suggested

While YOLOv8 offers state-of-the-art accuracy and efficiency, we chose not to use it for this project. The decision was influenced by several factors:

1. **Respect for Traditional Models:** Traditional machine learning models often provide a solid baseline and can be preferable in scenarios where model interpretability or computational resources are limited. These models are well-established and reliable in various applications.

2. **Specific Use Case Requirements:** Despite YOLOv8’s impressive performance, the specific requirements and constraints of our project aligned better with alternative approaches. Traditional models were selected to ensure robustness and flexibility for our particular use case.

3. **Computational Efficiency:** YOLOv8’s high computational demands may not be necessary for all scenarios. For environments with limited resources or where computational efficiency is a priority, traditional models can offer a practical solution.

## Benefits of YOLOv8

Though YOLOv8 was not chosen, its benefits include:

- **State-of-the-Art Performance:** YOLOv8 is renowned for its accuracy and speed. Its multi-scale architecture allows efficient detection of objects of various sizes.
- **Versatility:** It adapts well to a wide range of object detection tasks, including real-time video streams and aerial imagery.
- **Community Support:** With an active open-source community, YOLOv8 provides extensive resources, tutorials, and pre-trained models.
- **Efficiency:** YOLOv8 delivers excellent results even on resource-constrained devices.
- **Robustness:** It handles occlusions, diverse lighting conditions, and complex scenes effectively.

## Setup

1. **Install Ultralytics and Dependencies:**

   ```bash
   pip install ultralytics
   ```

   The `ultralytics` library is essential for working with YOLOv8 models.

2. **Check Software and Hardware:**

   After installing `ultralytics`, check your environment for compatibility:

   ```python
   import ultralytics
   ultralytics.checks()
   ```

   Verifying compatibility ensures efficient model performance and identifies any potential issues.

## Dataset Installation

1. **Install Roboflow and Download Dataset:**

   ```bash
   pip install roboflow
   ```

   ```python
   from roboflow import Roboflow

   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("hku-uas-deprecated-sobt2").project("standard_object_shape")
   dataset = project.version(2).download("yolov8")
   ```

   The dataset from Roboflow includes aerial images with various shapes, closely simulating real-world scenarios.

## Data Exploration

Explore and analyze the dataset to understand its composition:

```python
import os
import numpy as np
import matplotlib.pyplot as plt

train_dir = '/content/standard_object_shape-2/train'
test_dir = '/content/standard_object_shape-2/test'
val_dir = '/content/standard_object_shape-2/valid'

class_mapping = {
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

def explore_dataset(dataset_dir, dataset_name):
    num_images = len(os.listdir(os.path.join(dataset_dir, 'images')))
    num_labels = len(os.listdir(os.path.join(dataset_dir, 'labels')))

    print(f"Exploring {dataset_name} dataset:")
    print(f"Number of images: {num_images}")
    print(f"Number of labels (annotations): {num_labels}")

    class_counts = {}
    for label_file in os.listdir(os.path.join(dataset_dir, 'labels')):
        with open(os.path.join(dataset_dir, 'labels', label_file), 'r') as file:
            for line in file:
                class_name = line.strip().split()[0]
                class_label = class_mapping.get(class_name, class_name)
                class_counts[class_label] = class_counts.get(class_label, 0) + 1

    classes, counts = zip(*class_counts.items())
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.title(f'Class Distribution in {dataset_name} Dataset')
    plt.xlabel('Classes (Shapes)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

explore_dataset(train_dir, 'Training')
explore_dataset(test_dir, 'Testing')
explore_dataset(val_dir, 'Validation')
```

## Exploring Random Sample Images

View random sample images from the training dataset:

```python
import cv2
import random

sample_images = random.sample(os.listdir(os.path.join(train_dir, 'images')), 5)
plt.figure(figsize=(15, 5))
for i, image_filename in enumerate(sample_images):
    image = cv2.imread(os.path.join(train_dir, 'images', image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Sample Image {i+1}')
plt.show()
```

## Initializing the YOLOv8 Model

Initialize the YOLOv8 model with pre-trained weights:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

## Training the YOLOv8 Model

Train the model using the provided dataset:

```python
model.train(data="/content/standard_object_shape-2/data.yaml", epochs=3)
```

## Evaluating the YOLOv8 Model

### Confusion Matrix

Assess the model's predictions compared to the ground truth using a confusion matrix.

### Precision-Recall Curve

Evaluate the trade-off between precision and recall to determine the optimal prediction threshold.

### Confidence Curve

Analyze the model's confidence levels for its predictions.

## Performing Object Detection on Test Images

Perform detection on test images and save results:

```python
infer = YOLO("/content/runs/detect/train/weights/best.pt")
infer.predict("/content/standard_object_shape-2/test/images", save=True, save_txt=True)
```

### Predicted Images

Review visual results of object detection, showcasing how well the model detects various shapes.

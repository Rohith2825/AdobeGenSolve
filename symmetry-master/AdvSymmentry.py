import numpy as np
import bezier
import tensorflow as tf
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import cosine_similarity
from svgpathtools import svg2paths2
from shapely.geometry import Polygon, LineString
import xml.etree.ElementTree as ET
from io import BytesIO
from shapely.validation import make_valid

import matplotlib.pyplot as plt

def path_to_contour(path):
    contour = []
    for seg in path:
        for point in [seg.start, seg.end]:
            contour.append([int(point.real), int(point.imag)])
    return np.array(contour)

def process_svg(file_path):
    with open(file_path, 'rb') as file:
        file_data = file.read()
    paths, attributes, svg_attributes = svg2paths2(BytesIO(file_data))
    tree = ET.parse(BytesIO(file_data))
    root = tree.getroot()
    
    contours = []
    for path in paths:
        contour = path_to_contour(path)
        if contour.size != 0:
            contours.append(contour)
    return contours

def build_symmetry_net(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_points(points, target_shape=(64, 64)):
    """
    Preprocess points to fit the model input shape.
    Args:
        points: Original points array
        target_shape: Desired shape for the model
    Returns:
        Processed points array
    """
    # Assuming points are in shape (num_points, 2)
    # Normalize points to fit within the target shape
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    points_range = points_max - points_min
    
    # Create an empty image of target shape
    image = np.zeros(target_shape)
    
    # Scale points to fit within the target shape
    scaled_points = (points - points_min) / points_range
    scaled_points = (scaled_points * (target_shape[0] - 1)).astype(int)
    
    # Populate the image with points
    for point in scaled_points:
        x, y = point
        if 0 <= x < target_shape[0] and 0 <= y < target_shape[1]:
            image[x, y] = 1  # Set pixel to 1
    
    # Add a channel dimension if necessary
    image = np.expand_dims(image, axis=-1)
    
    return image

def detect_symmetry_net(contours, model):
    symmetric_scores = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 8:
            continue
        
        # Preprocess points to fit the model input shape
        processed_points = preprocess_points(points)
        processed_points = np.expand_dims(processed_points, axis=0)  # Add batch dimension
        scores = model.predict(processed_points)
        symmetric_scores.append(np.mean(scores))
    
    return symmetric_scores


def detect_symmetry_loy_eklundh(contours):
    symmetric_pairs = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        
        centroid = np.mean(points, axis=0)
        distances = dist.cdist(points, [centroid])
        distances = distances.reshape(-1)
        distances = np.sort(distances)
        symmetric_score = np.sum(np.abs(distances - np.mean(distances))) / len(distances)
        
        symmetric_pairs.append((points, symmetric_score))
    
    return symmetric_pairs

def fit_bezier(points, degree=None):
    if degree is None:
        degree = len(points) - 1
    nodes = np.asfortranarray(points.T)
    curve = bezier.Curve(nodes, degree=degree)
    return curve

def bezier_curves_from_symmetry(symmetry_pairs):
    bezier_curves = []
    for points, _ in symmetry_pairs:
        degree = len(points) - 1
        curve = fit_bezier(points, degree)
        bezier_curves.append(curve)
    return bezier_curves

def build_gan_models(input_shape):
    def build_generator():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=100, activation='relu'),
            tf.keras.layers.Dense(np.prod(input_shape), activation='tanh'),
            tf.keras.layers.Reshape(input_shape)
        ])
        return model

    def build_discriminator():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_gan(generator, discriminator):
        discriminator.trainable = False
        model = tf.keras.Sequential([generator, discriminator])
        return model

    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return gan, generator, discriminator

def detect_symmetry_gan(contours, gan_model, generator):
    symmetric_scores = []
    for contour in contours:
        # Generate a synthetic contour from random noise
        noise = np.random.normal(0, 1, (1, 100))
        generated_contour = generator.predict(noise)[0]
        generated_contour = np.clip(generated_contour, 0, 255)  # Assuming image pixel range

        # Reshape contour to match the dimensions of generated_contour if necessary
        contour_reshaped = np.resize(contour.flatten(), generated_contour.flatten().shape)
        
        # Calculate similarity between generated and real contour
        similarity = cosine_similarity([contour_reshaped], [generated_contour.flatten()])
        score = np.mean(similarity)
        symmetric_scores.append(score)
    return symmetric_scores

def find_lines_of_symmetry(contour):
    lines = []
    
    if len(contour) < 4:
        print("Contour has insufficient points for symmetry detection.")
        return lines

    try:
        contour_np = np.array(contour)
        contour_polygon = Polygon(contour_np)
        
        contour_polygon = make_valid(contour_polygon)
        
        if not contour_polygon.is_valid:
            print("Contour polygon is invalid after making it valid.")
            return lines

        centroid = contour_np.mean(axis=0)
        distances = np.linalg.norm(contour_np - centroid, axis=1)
        avg_distance = np.mean(distances)

        # Define tolerance for symmetry detection
        tolerance = avg_distance * 0.1

        vertical_line = LineString([(contour_np[:, 0].min(), 0), (contour_np[:, 0].max(), 0)])
        if contour_polygon.symmetric_difference(vertical_line).area < tolerance:
            lines.append('Vertical')

        horizontal_line = LineString([(0, contour_np[:, 1].min()), (0, contour_np[:, 1].max())])
        if contour_polygon.symmetric_difference(horizontal_line).area < tolerance:
            lines.append('Horizontal')

        diagonal_line1 = LineString([(contour_np[:, 0].min(), contour_np[:, 1].min()), (contour_np[:, 0].max(), contour_np[:, 1].max())])
        if contour_polygon.symmetric_difference(diagonal_line1).area < tolerance:
            lines.append('Diagonal (45 degrees)')

        diagonal_line2 = LineString([(contour_np[:, 0].min(), contour_np[:, 1].max()), (contour_np[:, 0].max(), contour_np[:, 1].min())])
        if contour_polygon.symmetric_difference(diagonal_line2).area < tolerance:
            lines.append('Diagonal (-45 degrees)')
    
    except Exception as e:
        print(f"Error processing contour: {e}")
    
    return lines

def visualize_contours_and_symmetry(file_path):
    contours = process_svg(file_path)
    for contour in contours:
        plt.figure(figsize=(8, 8))
        plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Contour')
        
        lines_of_symmetry = find_lines_of_symmetry(contour)
        for line in lines_of_symmetry:
            if line == 'Vertical':
                plt.axvline(x=np.mean(contour[:, 0]), color='r', linestyle='--', label='Vertical Symmetry')
            elif line == 'Horizontal':
                plt.axhline(y=np.mean(contour[:, 1]), color='g', linestyle='--', label='Horizontal Symmetry')
            elif line == 'Diagonal (45 degrees)':
                plt.plot([contour[:, 0].min(), contour[:, 0].max()], 
                         [contour[:, 1].min(), contour[:, 1].max()], 
                         'c--', label='Diagonal (45 degrees)')
            elif line == 'Diagonal (-45 degrees)':
                plt.plot([contour[:, 0].min(), contour[:, 0].max()], 
                         [contour[:, 1].max(), contour[:, 1].min()], 
                         'm--', label='Diagonal (-45 degrees)')
        
        plt.legend()
        plt.title('Contour and Symmetry Lines Visualization')
        plt.show()

def evaluate_symmetry_and_fit(contours, model, gan_model, generator, threshold=0.5):
    symmetry_scores_net = detect_symmetry_net(contours, model)
    symmetry_scores_loy_eklundh = detect_symmetry_loy_eklundh(contours)
    symmetry_scores_gan = detect_symmetry_gan(contours, gan_model, generator)
    
    bezier_curves = bezier_curves_from_symmetry(symmetry_scores_loy_eklundh)
    
    best_symmetry_score_net = max(symmetry_scores_net) if symmetry_scores_net else 0
    best_symmetry_score_loy_eklundh = max([score for _, score in symmetry_scores_loy_eklundh]) if symmetry_scores_loy_eklundh else 0
    best_symmetry_score_gan = max(symmetry_scores_gan) if symmetry_scores_gan else 0
    
    decision_tree = {
        'net': best_symmetry_score_net > threshold,
        'loy_eklundh': best_symmetry_score_loy_eklundh < threshold,
        'gan': best_symmetry_score_gan > threshold,
    }
    
    final_decision = decision_tree['net'] or decision_tree['loy_eklundh'] or decision_tree['gan']
    
    print("Symmetry scores (NN, Loy-Eklundh, GAN):", best_symmetry_score_net, best_symmetry_score_loy_eklundh, best_symmetry_score_gan)
    print("Final decision on symmetry:", final_decision)
    
    return final_decision, bezier_curves

# Sample file path for testing
file_path = 'C:/Users/rohit/Downloads/symmetry-master/symmetry-master/test/butterfly-svgrepo-com.svg'

# Build and train the models
input_shape = (64, 64, 1)  # Adjust this according to your needs
symmetry_net = build_symmetry_net(input_shape)
gan, generator, discriminator = build_gan_models(input_shape)

# Process and visualize the SVG file
visualize_contours_and_symmetry(file_path)

# Evaluate symmetry and fit Bezier curves
contours = process_svg(file_path)
final_decision, bezier_curves = evaluate_symmetry_and_fit(contours, symmetry_net, gan, generator)


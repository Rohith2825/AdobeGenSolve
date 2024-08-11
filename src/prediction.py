import numpy as np
import cv2
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an image for prediction.
    
    Parameters:
    - image_path: Path to the image file.
    - target_size: Size to which the image should be resized.
    
    Returns:
    - img: Preprocessed image as a numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if(img is not None):
        img = cv2.resize(img, target_size)
        img = img.reshape((1, 128, 128, 1)) / 255.0
        
    return img

def predict_image(model, image_path, label_encoder):
    """ 
    Predict the class of an image.
    
    Parameters:
    - model: The trained model to use for prediction.
    - image_path: Path to the image file.
    - label_encoder: The label encoder used to encode the classes.
    
    Returns:
    - predicted_class: The predicted class of the image.
    """
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

def display_image_with_prediction(image_path, predicted_class):
    """
    Display an image with its predicted class as the window title.
    
    Parameters:
    - image_path: Path to the image file.
    - predicted_class: The predicted class of the image.
    
    Returns:
    - None
    """
    img = cv2.imread(image_path)
    cv2.imshow(f'{predicted_class}', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    

# Loading the Label Encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([  'circle', 'heptagon','hexagon','nonagon','octagon','pentagon','square',  'star','triangle' ])

# Loading the trained model
model = load_model("C:/Users/rohit/Downloads/symmetry-master/dectection-master/models/identifier.h5")


# Test Prediction
user_image_path = 'C:/Users/rohit/Downloads/symmetry-master/starDoodle.png'
predicted_class = predict_image(model, user_image_path, label_encoder)
print(f'Predicted Class: {predicted_class}')
display_image_with_prediction(user_image_path, predicted_class)


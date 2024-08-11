import os
import shutil

# Define the folder paths
source_folder = 'C:/Users/rohit/Downloads/symmetry-master/detection-master/output'
destination_folder = 'C:/Users/rohit/Downloads/symmetry-master/detection-master/sortedSet'

# Shape classes
shapes = ['Circle', 'Triangle', 'Square', 'Pentagon', 'Hexagon', 'Heptagon', 'Octagon', 'Nonagon', 'Star']

# Create folders for each shape class
for shape in shapes:
    shape_folder = os.path.join(destination_folder, shape)
    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)

# Iterate through images and move them to relevant folders
for filename in os.listdir(source_folder):
    for shape in shapes:
        if shape in filename:
            shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, shape))
            break  # Exit the loop once the image is moved

print("Images sorted into folders successfully.")

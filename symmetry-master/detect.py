# import sys
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Initialize SIFT detector
# sift = cv2.SIFT_create()

# def very_close(a, b, tol=4.0):
#     return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol

# def S(si, sj, sigma=1):
#     q = (-abs(si - sj)) / (sigma * (si + sj))
#     return np.exp(q ** 2)

# def reisfeld(phi, phj, theta):
#     return 1 - np.cos(phi + phj - 2 * theta)

# def midpoint(i, j):
#     return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2

# def angle_with_x_axis(i, j):
#     x, y = i[0] - j[0], i[1] - j[1]
#     if x == 0:
#         return np.pi / 2
#     angle = np.arctan(y / x)
#     if angle < 0:
#         angle += np.pi
#     return angle


# def draw(image, r, theta, line_thickness=2, line_color=(255, 101, 35)):
#     height, width = image.shape

#     if np.pi / 4 < theta < 3 * (np.pi / 4):
#         # Vertical line
#         for x in range(width):
#             y = int((r - x * np.cos(theta)) / np.sin(theta))
#             if 0 <= y < height:
#                 image[y, x] = line_color[0]
#     else:
#         # Horizontal line
#         for y in range(height):
#             x = int((r - y * np.sin(theta)) / np.cos(theta))
#             if 0 <= x < width:
#                 image[y, x] = line_color[0]

#     # Draw line with thickness and color
#     line_color_bgr = (line_color[0], line_color[1], line_color[2])  # BGR format for OpenCV
#     if np.pi / 4 < theta < 3 * (np.pi / 4):
#         cv2.line(image, (0, int((r - 0 * np.cos(theta)) / np.sin(theta))),
#                  (width - 1, int((r - (width - 1) * np.cos(theta)) / np.sin(theta))),
#                  line_color_bgr, thickness=line_thickness)
#     else:
#         cv2.line(image, (int((r - 0 * np.sin(theta)) / np.cos(theta)), 0),
#                  (int((r - (height - 1) * np.sin(theta)) / np.cos(theta)), height - 1),
#                  line_color_bgr, thickness=line_thickness)




# def superm2(image):
#     """Performs the symmetry detection on image and automatically selects the best line of symmetry."""
#     mimage = np.fliplr(image)
#     kp1, des1 = sift.detectAndCompute(image, None)
#     kp2, des2 = sift.detectAndCompute(mimage, None)
    
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good_matches = []
#     houghr = []
#     houghth = []
#     weights = []

#     for match, match2 in matches:
#         point = kp1[match.queryIdx]
#         mirpoint = kp2[match.trainIdx]
#         mirpoint2 = kp2[match2.trainIdx]
#         mirpoint2.angle = np.pi - mirpoint2.angle
#         mirpoint.angle = np.pi - mirpoint.angle
#         if mirpoint.angle < 0.0:
#             mirpoint.angle += 2 * np.pi
#         if mirpoint2.angle < 0.0:
#             mirpoint2.angle += 2 * np.pi
#         mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])

#         if very_close(point.pt, mirpoint.pt):
#             mirpoint = mirpoint2
#             good_matches.append(match2)
#         else:
#             good_matches.append(match)

#         theta = angle_with_x_axis(point.pt, mirpoint.pt)
#         xc, yc = midpoint(point.pt, mirpoint.pt)
#         r = xc * np.cos(theta) + yc * np.sin(theta)
#         Mij = reisfeld(point.angle, mirpoint.angle, theta) * S(point.size, mirpoint.size)

#         houghr.append(r)
#         houghth.append(theta)
#         weights.append(Mij)

#     # Convert lists to numpy arrays
#     houghr = np.array(houghr)
#     houghth = np.array(houghth)
#     weights = np.array(weights)

#     # Generate hexbin plot and find the bin with the maximum count
#     plt.figure(figsize=(8, 6))
#     hexbin = plt.hexbin(houghr, houghth, bins=200, gridsize=50, cmap='inferno')
#     counts = hexbin.get_array()
    
#     # Get the bin with the maximum count
#     max_bin_idx = counts.argmax()
    
#     # Extract the coordinates (r, theta) corresponding to the bin with the maximum count
#     r_bins = hexbin.get_offsets()[:, 0]
#     theta_bins = hexbin.get_offsets()[:, 1]
#     best_r = r_bins[max_bin_idx]
#     best_theta = theta_bins[max_bin_idx]
    
#     print(f"Best symmetry line detected at r={best_r}, theta={best_theta}")

#     # Draw the line of symmetry on the image
#     draw(image, best_r, best_theta)
#     cv2.imshow("Symmetry Detection", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python detect.py IMAGE")
#         return

#     image = cv2.imread(sys.argv[1], 0)
#     superm2(image)

# if __name__ == "__main__":
#     main()

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

def very_close(a, b, tol=4.0):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol

def S(si, sj, sigma=1):
    q = (-abs(si - sj)) / (sigma * (si + sj))
    return np.exp(q ** 2)

def reisfeld(phi, phj, theta):
    return 1 - np.cos(phi + phj - 2 * theta)

def midpoint(i, j):
    return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2

def angle_with_x_axis(i, j):
    x, y = i[0] - j[0], i[1] - j[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

# def draw(image, r, theta, line_thickness=2, line_color=(255, 101, 35)):
#     height, width = image.shape

#     if np.pi / 4 < theta < 3 * (np.pi / 4):
#         # Vertical line
#         for x in range(width):
#             y = int((r - x * np.cos(theta)) / np.sin(theta))
#             if 0 <= y < height:
#                 image[y, x] = line_color[0]
#     else:
#         # Horizontal line
#         for y in range(height):
#             x = int((r - y * np.sin(theta)) / np.cos(theta))
#             if 0 <= x < width:
#                 image[y, x] = line_color[0]

#     # Draw line with thickness and color
#     line_color_bgr = (line_color[0], line_color[1], line_color[2])  # BGR format for OpenCV
#     if np.pi / 4 < theta < 3 * (np.pi / 4):
#         cv2.line(image, (0, int((r - 0 * np.cos(theta)) / np.sin(theta))),
#                  (width - 1, int((r - (width - 1) * np.cos(theta)) / np.sin(theta))),
#                  line_color_bgr, thickness=line_thickness)
#     else:
#         cv2.line(image, (int((r - 0 * np.sin(theta)) / np.cos(theta)), 0),
#                  (int((r - (height - 1) * np.sin(theta)) / np.cos(theta)), height - 1),
#                  line_color_bgr, thickness=line_thickness)


def draw(image, r, theta, line_thickness=2, line_color=(255, 101, 35)):
    # Check if image is grayscale or color
    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Unexpected image shape")

    if np.pi / 4 < theta < 3 * (np.pi / 4):
        # Vertical line
        for x in range(width):
            y = int((r - x * np.cos(theta)) / np.sin(theta))
            if 0 <= y < height:
                image[y, x] = line_color[0]
    else:
        # Horizontal line
        for y in range(height):
            x = int((r - y * np.sin(theta)) / np.cos(theta))
            if 0 <= x < width:
                image[y, x] = line_color[0]

    # Draw line with thickness and color
    line_color_bgr = (line_color[0], line_color[1], line_color[2])  # BGR format for OpenCV
    if np.pi / 4 < theta < 3 * (np.pi / 4):
        cv2.line(image, (0, int((r - 0 * np.cos(theta)) / np.sin(theta))),
                 (width - 1, int((r - (width - 1) * np.cos(theta)) / np.sin(theta))),
                 line_color_bgr, thickness=line_thickness)
    else:
        cv2.line(image, (int((r - 0 * np.sin(theta)) / np.cos(theta)), 0),
                 (int((r - (height - 1) * np.sin(theta)) / np.cos(theta)), height - 1),
                 line_color_bgr, thickness=line_thickness)


def superm2(image):
    """Performs the symmetry detection on image and automatically selects the best line of symmetry."""
    mimage = np.fliplr(image)
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(mimage, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    houghr = []
    houghth = []
    weights = []

    for match, match2 in matches:
        point = kp1[match.queryIdx]
        mirpoint = kp2[match.trainIdx]
        mirpoint2 = kp2[match2.trainIdx]
        mirpoint2.angle = np.pi - mirpoint2.angle
        mirpoint.angle = np.pi - mirpoint.angle
        if mirpoint.angle < 0.0:
            mirpoint.angle += 2 * np.pi
        if mirpoint2.angle < 0.0:
            mirpoint2.angle += 2 * np.pi
        mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])

        if very_close(point.pt, mirpoint.pt):
            mirpoint = mirpoint2
            good_matches.append(match2)
        else:
            good_matches.append(match)

        theta = angle_with_x_axis(point.pt, mirpoint.pt)
        xc, yc = midpoint(point.pt, mirpoint.pt)
        r = xc * np.cos(theta) + yc * np.sin(theta)
        Mij = reisfeld(point.angle, mirpoint.angle, theta) * S(point.size, mirpoint.size)

        houghr.append(r)
        houghth.append(theta)
        weights.append(Mij)

    # Convert lists to numpy arrays
    houghr = np.array(houghr)
    houghth = np.array(houghth)
    weights = np.array(weights)

    # Generate hexbin plot and find the bin with the maximum count
    plt.figure(figsize=(8, 6))
    hexbin = plt.hexbin(houghr, houghth, bins=200, gridsize=50, cmap='inferno')
    counts = hexbin.get_array()
    
    # Get the bin with the maximum count
    max_bin_idx = counts.argmax()
    
    # Extract the coordinates (r, theta) corresponding to the bin with the maximum count
    r_bins = hexbin.get_offsets()[:, 0]
    theta_bins = hexbin.get_offsets()[:, 1]
    best_r = r_bins[max_bin_idx]
    best_theta = theta_bins[max_bin_idx]
    
    print(f"Best symmetry line detected at r={best_r}, theta={best_theta}")

    # Draw the line of symmetry on the image
    draw(image, best_r, best_theta)

    # Convert BGR image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Symmetry Detection")
    plt.axis('off')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect.py IMAGE")
        return

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Error: Image not found or unable to read.")
        return

    superm2(image)

if __name__ == "__main__":
    main()

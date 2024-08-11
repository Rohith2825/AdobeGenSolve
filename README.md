# Adobe-GenSolve
## CURVETOPIA

### Introduction

Welcome to **CURVETOPIA: A Journey into the World of Curves**. This project explores the fascinating realm of 2D curves, focusing on their identification, regularization, and beautification in Euclidean space. Our work addresses three primary challenges in computational geometry and computer vision: curve regularization, symmetry detection, and curve completion.

### Objective

Our mission is to develop a robust system that can process 2D curves, transforming raw input into well-defined, aesthetically pleasing geometric forms. We aim to create an end-to-end process that takes line art as input and outputs a set of curves defined as connected sequences of cubic Bezier curves.

### Problem Statement

While our ultimate goal is to work with PNG (raster) images of line art, we've started with a simplified approach. Our current input consists of polylines, defined as sequences of points in \( \mathbb{R}^2 \). Specifically, we work with a finite subset of paths from \( P \), where \( P \) is the set of all paths in \( \mathbb{R}^2 \). Each path is a finite sequence of points \(\{p_i\}_{1 \leq i \leq n}\) from \( \mathbb{R}^2 \).

Our task is to transform this input into another set of paths that exhibit the desired properties of regularization, symmetry, and completeness. For visualization purposes, we use the SVG format, which can be rendered in a browser, with the output curves represented as cubic Bézier curves.

### Key Challenges

- **Curve Regularization:** We identify and regularize various shapes within the given set of curves. This includes detecting and refining primitives such as straight lines, circles, ellipses, rectangles (including rounded rectangles), regular polygons, and star shapes.

- **Symmetry Exploration:** For closed shapes, we detect the presence of symmetry, focusing primarily on reflection symmetries. This involves identifying lines of symmetry where the shape can be divided into mirrored halves.

- **Curve Completion:** We address the challenge of completing curves that have been "planarized" due to overlapping portions being removed. This task requires us to naturally complete curves with gaps or partial holes.


## Table of Contents

1. [Shapes Detection](#shapes-detection)
   - [Detection of Various Shapes](#detection-of-various-shapes)
   - [Custom Model Development](#custom-model-development)
2. [Symmetry Detection](#symmetry-detection)
   - [Loy-Eklundh Algorithm](#loy-eklundh-algorithm)
   - [Symmetry Net](#symmetry-net)
   - [GAN for Symmetry Detection](#gan-for-symmetry-detection)
   - [Bezier Curves](#bezier-curves)
3. [Curve Completion](#curve-completion)
   - [DeepFill Algorithm](#deepfill-algorithm)
   - [3D Conversion Occlusion Model](#3d-conversion-occlusion-model)
   - [Acknowledgements](#acknowledgements)



## Shapes Detection

### Detection of Various Shapes

This task involves identifying and classifying regular shapes within curves. The target use cases are hand-drawn shapes and doodles. Here are the specific shapes we aim to detect:

1. **Straight Lines:** Recognize and classify straight lines within images.
2. **Circles and Ellipses:** Identify circles, which are equidistant from a center, and ellipses, which have two focal points.
3. **Rectangles and Rounded Rectangles:** Distinguish between standard rectangles and those with rounded edges.
4. **Regular Polygons:** Detect polygons with equal sides and angles.
5. **Star Shapes:** Identify star shapes by detecting a central point with multiple radial arms.

### Custom Model Development

Initially, we used a pre-trained YOLOv8 model, known for its high accuracy in detecting shapes from aerial views. However, for educational purposes and to enhance learning, we developed a custom Convolutional Neural Network (CNN). This model was trained on approximately 90,000 images of different shapes. 

While the YOLOv8 model offered high accuracy, the custom CNN allows us to delve deeper into the specifics of shape detection and provides a learning experience. The accuracy of our custom model can be further improved with a larger and more diverse dataset.

**Special Thanks:** Thanks to Adobe for providing the opportunity to explore and innovate in the realm of shapes detection.

## Symmetry Detection

### Loy-Eklundh Algorithm

The Loy-Eklundh algorithm is a classical approach to symmetry detection. It analyzes contour distances and their distribution to identify potential symmetry lines. This method provides foundational insights into how symmetry works.

**Visual Demonstration:** The algorithm detects symmetry lines based on the geometric properties of contours.

### Symmetry Net

Symmetry Net is a neural network-based model designed to predict symmetry by processing contour points. This deep learning approach is trained to recognize patterns and assess symmetry accurately.

**Visual Demonstration:** Observe how Symmetry Net evaluates contours and provides symmetry scores.

### GAN for Symmetry Detection

Generative Adversarial Networks (GANs) are employed to generate synthetic contours for comparison. The GAN model generates symmetrical contours, which are compared with input contours to assess the degree of symmetry.

**Visual Demonstration:** See how GANs evaluate and generate symmetrical contours.

### Bezier Curves

Bezier curves are applied to contour data to help visualize symmetrical properties. This technique adjusts curves to match symmetrical patterns, providing insights into the symmetry of shapes.

**Visual Demonstration:** Explore how Bezier curves assist in visualizing symmetry within contours.

## Curve Completion

### DeepFill Algorithm

To address curve completion, we utilize the DeepFill algorithm. This method fills gaps in curves caused by occlusions, guided by smoothness, regularity, and symmetry.

**Reference:**
- **Free-Form Image Inpainting with Gated Convolution**
- **Authors:** Yu, Jiahui; Lin, Zhe; Yang, Jimei; Shen, Xiaohui; Lu, Xin; Huang, Thomas S
- **Year:** 2018
- **Link:** [Original Paper](https://arxiv.org/abs/1806.03589)

### 3D Conversion Occlusion Model

We propose enhancing the DeepFill algorithm with a 3D conversion occlusion model to improve performance. This model aims to handle complex occlusions better by incorporating 3D data.

**Future Work:** We plan to explore new LLM-based learning approaches to further enhance the model’s capabilities.

### Acknowledgements

We acknowledge the foundational contributions of the authors of the DeepFill algorithm, whose work has significantly influenced our approach to curve completion.

## Contact

For questions or contributions, please reach out to us at [rohith2210194@ssn.edu.in].


Have a look through our demo here,
<video controls src="AdobeDEMO - Made with Clipchamp.mp4" title="Title"></video>
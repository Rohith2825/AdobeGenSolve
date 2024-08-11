# Loy-Eklundh Symmetry Detection

## About

This line of symmetry detector was developed as a final project for the CS39R Symmetry and Topology seminar in Spring 2016. The tool identifies a line of reflective symmetry in an image using the algorithm proposed by Loy and Eklundh in their research paper:

[Detecting Symmetry and Symmetric Constellations of Features](http://www.cse.psu.edu/~yul11/CourseFall2006_files/loy_eccv2006.pdf) by Loy and Eklundh.

### Adobe GenSolve

This project is brought to you by a dedicated team from SSN College of Engineering (SSNCE), who are passionate about traditional algorithms and their applications. Our commitment to exploring and implementing foundational techniques drives our work, and this project exemplifies our enthusiasm for leveraging classical methods in the realm of computer vision.

## To Run

You need OpenCV, Matplotlib, and NumPy to run the script. If not already installed, you can install them using:

```bash
pip install opencv-python-headless matplotlib numpy
```

## How to Use

1. **Run the Detection:**

   Execute the script with your image file:

   ```bash
   python detect.py your_image.png
   ```

   The script will automatically detect the line of symmetry and display the result.

2. **Save the Output:**

   The processed image with the detected line of symmetry will be saved automatically as a JPEG file in the same directory.

## Mathematical Details

### Key Formulas

1. **Symmetry Function \( S \):**

   Measures the similarity between the sizes of feature points:
   \[
   S(si, sj, \sigma) = \exp \left( \frac{-(\text{abs}(si - sj))^2}{\sigma^2 (si + sj)^2} \right)
   \]

2. **Reisfeld Function \( M \):**

   Evaluates the match between angles of feature points:
   \[
   M(\phi, \phi_j, \theta) = 1 - \cos(\phi + \phi_j - 2 \theta)
   \]

3. **Midpoint Function:**

   Computes the midpoint between two points:
   \[
   \text{midpoint}(i, j) = \left( \frac{i[0] + j[0]}{2}, \frac{i[1] + j[1]}{2} \right)
   \]

4. **Angle with X-Axis:**

   Determines the angle between two points and the x-axis:
   \[
   \text{angle} = \text{atan2}(y, x)
   \]

### Algorithmic Approach

The primary goal of our symmetry detection algorithm is to identify a line of reflective symmetry within an image. While traditional methods focus on detecting symmetry along predefined axes—such as horizontal, vertical, or diagonal—our approach extends beyond these constraints.

**Our Algorithm's Capabilities:**

- **Flexible Symmetry Detection:** Unlike conventional methods that limit detection to specific orientations, our algorithm is designed to find a line of symmetry in any direction. It works with a wide range of possible symmetry lines, including those that are not strictly horizontal, vertical, or diagonal.
- **Automatic Detection:** The script automatically determines the optimal line of symmetry based on feature matching and symmetry voting, removing the need for manual input of parameters such as angle and distance.

### Why a Custom Solution?

While extensive libraries and frameworks offer sophisticated tools for image processing, our handmade solution stands out for several reasons:

1. **Educational Value:** Developing this custom tool enhances understanding of the fundamental algorithms and mathematical principles behind symmetry detection, offering insights that go beyond high-level abstractions.

2. **Flexibility:** Tailoring the solution to specific project needs allows for unique optimizations and customizations that may not be available in general-purpose libraries.

3. **Control:** A custom implementation provides full control over the detection process, facilitating precise adjustments and improvements based on experimental observations and specific requirements.

4. **Insight into Algorithms:** Implementing this algorithm from scratch not only achieves the goal of detecting lines of symmetry but also offers a deeper appreciation of feature matching and symmetry detection techniques.

In summary, this project reflects our team’s dedication to traditional algorithms and showcases the value of applying classical methods to modern challenges in computer vision.

import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np

def extract_walls(image):
    """
    Extract wall-like structures from the SLAM map.
    """
    # Invert image
    inverted = cv2.bitwise_not(image)

    # Threshold to create binary image
    _, binary = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Denoise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Detect edges
    edges = cv2.Canny(cleaned, 50, 150)

    return cleaned, edges

def save_image(image, path):
    cv2.imwrite(path, image)


def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_map_and_metadata(pgm_path, yaml_path):
    # Load PGM image
    image = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

    # Load YAML metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    return image, metadata

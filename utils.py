import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np

def extract_walls(image):
    """
    Enhanced wall extraction with preprocessing to connect fragmented structures.
    """
    # Invert image: walls = white, free space = black
    inverted = cv2.bitwise_not(image)

    # Threshold to keep only high-contrast wall areas
    _, binary = cv2.threshold(inverted, 220, 255, cv2.THRESH_BINARY)

    # Morphological closing to connect nearby wall fragments
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Blurring for noise
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return closed, edges


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

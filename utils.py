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

def detect_lines(edge_image, min_line_length=50, max_line_gap=10):
    """
    Detect straight lines using the Probabilistic Hough Transform.
    """
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return lines

def draw_lines(image_shape, lines, color=(255, 255, 255), thickness=2):
    """
    Draw lines on a blank canvas.
    """
    line_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def extract_wall_contours(binary_image, min_area=500):
    """
    Find large contours (wall-like blobs) in the binary mask.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_img = np.zeros_like(binary_image)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(contour_img, [contour], -1, 255, thickness=cv2.FILLED)
    return contour_img

def polygonize_contours(binary_image, min_area=500, epsilon_factor=0.01):
    """
    Approximate wall contours as polygons (connected straight segments).
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygon_img = np.zeros_like(binary_image)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Draw polygon with filled straight segments
            cv2.polylines(polygon_img, [approx], isClosed=True, color=255, thickness=2)
            cv2.fillPoly(polygon_img, [approx], 255)

    return polygon_img



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

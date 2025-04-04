import yaml
import cv2
import numpy as np
from shapely.geometry import LineString

def extend_line(pt1, pt2, length=50):
    """
    Extend a line segment beyond both ends.
    """
    vec = np.array(pt2) - np.array(pt1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return pt1, pt2

    unit = vec / norm
    extended_pt1 = tuple((np.array(pt1) - unit * length).astype(int))
    extended_pt2 = tuple((np.array(pt2) + unit * length).astype(int))
    return extended_pt1, extended_pt2

def connect_intersecting_lines(lines_img, min_intersection_dist=20, extend_length=30):
    """
    Detect endpoints that are near intersection, and connect them.
    """
    # Detect original lines
    edges = cv2.Canny(lines_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    if lines is None:
        return lines_img

    extended_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pt1, pt2 = extend_line((x1, y1), (x2, y2), length=extend_length)
        extended_lines.append((pt1, pt2))

    intersections = []
    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            l1 = LineString(extended_lines[i])
            l2 = LineString(extended_lines[j])
            if l1.intersects(l2):
                p = l1.intersection(l2)
                if p.geom_type == 'Point':
                    intersections.append((int(p.x), int(p.y)))

    result = lines_img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for inter in intersections:
            for pt in [(x1, y1), (x2, y2)]:
                if np.linalg.norm(np.array(pt) - np.array(inter)) < min_intersection_dist:
                    cv2.line(result, pt, inter, 255, thickness=2)

    return result

def extract_walls(image):
    inverted = cv2.bitwise_not(image)
    _, binary = cv2.threshold(inverted, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return closed, edges

def detect_lines(edge_image, min_line_length=5, max_line_gap=150):
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return lines

def extract_wall_contours(binary_image, min_area=500):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(binary_image)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(contour_img, [contour], -1, 255, thickness=cv2.FILLED)
    return contour_img

def extract_wall_segments_as_lines(binary_image, min_area=100, epsilon_factor=0.01):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_img = np.zeros(binary_image.shape, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            for i in range(len(approx) - 1):
                pt1 = tuple(approx[i][0])
                pt2 = tuple(approx[i + 1][0])
                cv2.line(line_img, pt1, pt2, 255, thickness=2)
    return line_img

def save_image(image, path):
    cv2.imwrite(path, image)

def load_map_and_metadata(pgm_path, yaml_path):
    image = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return image, metadata

import yaml
import cv2
import numpy as np
from shapely.geometry import LineString, Point

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

import numpy as np
import cv2
from shapely.geometry import LineString, Point

import numpy as np
import cv2
from shapely.geometry import LineString

import numpy as np
import cv2
from shapely.geometry import LineString, Point

def extend_line(pt1, pt2, length=50):
    """
    Extend a line segment in both directions by 'length' while preserving direction.
    """
    pt1, pt2 = np.array(pt1), np.array(pt2)
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm == 0:
        return tuple(pt1), tuple(pt2)
    unit_vec = vec / norm
    ext_pt1 = tuple((pt1 - unit_vec * length).astype(int))
    ext_pt2 = tuple((pt2 + unit_vec * length).astype(int))
    return ext_pt1, ext_pt2

def connect_intersecting_lines(lines, lines_img, min_intersection_dist=20, extend_length=30):
    """
    Extend Hough lines and connect:
    - Endpoints to nearby intersection points
    - Endpoints to nearby endpoints (gap closing)
    Only connect isolated endpoints, i.e., those not already intersecting other lines.
    """
    result = lines_img.copy()

    if lines is None:
        edges = cv2.Canny(lines_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
        if lines is None:
            return result

    # Convert lines to extended format
    extended_lines = []
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pt1, pt2 = (x1, y1), (x2, y2)
        ext_pt1, ext_pt2 = extend_line(pt1, pt2, length=extend_length)
        extended_lines.append((ext_pt1, ext_pt2))
        endpoints.extend([pt1, pt2])

    # Prepare shapely segments for intersection & proximity testing
    line_segments = [LineString([pt1, pt2]) for pt1, pt2 in extended_lines]

    # Compute intersection points
    intersections = []
    for i in range(len(line_segments)):
        for j in range(i + 1, len(line_segments)):
            if line_segments[i].intersects(line_segments[j]):
                p = line_segments[i].intersection(line_segments[j])
                if p.geom_type == 'Point':
                    intersections.append((int(p.x), int(p.y)))

    # === Connect endpoints to intersection points (only isolated ones)
    for (x1, y1, x2, y2) in lines[:, 0]:
        for inter in intersections:
            inter_point = Point(inter)
            for pt in [(x1, y1), (x2, y2)]:
                pt_point = Point(pt)
                # Skip if this pt is already close to any other wall
                is_isolated = all(
                    segment.distance(pt_point) > (min_intersection_dist / 2)
                    for segment in line_segments
                )
                if is_isolated and pt_point.distance(inter_point) < min_intersection_dist:
                    cv2.line(result, pt, inter, 255, thickness=2)

    # === Also connect nearby endpoints directly (gap closers)
    for i, pt1 in enumerate(endpoints):
        for j, pt2 in enumerate(endpoints):
            if i < j and np.linalg.norm(np.array(pt1) - np.array(pt2)) < min_intersection_dist:
                cv2.line(result, pt1, pt2, 255, thickness=2)

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

def connect_line_endpoints(lines, image_shape, max_distance=100):
    """
    Connects line endpoints that are close to each other.
    """
    if lines is None:
        return np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Collect all endpoints
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.extend([(x1, y1), (x2, y2)])

    # Create canvas and draw original lines
    canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Connect nearby endpoints
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            pt1 = endpoints[i]
            pt2 = endpoints[j]
            if np.linalg.norm(np.array(pt1) - np.array(pt2)) < max_distance:
                cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    return canvas



def save_image(image, path):
    cv2.imwrite(path, image)

def load_map_and_metadata(pgm_path, yaml_path):
    image = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return image, metadata

def draw_lines(image_shape, lines, color=(255, 255, 255), thickness=2):
    """
    Draw lines on a blank canvas.
    """
    line_image = np.zeros(binary_image.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, pt1, pt2, 255, thickness=2)
    return line_image
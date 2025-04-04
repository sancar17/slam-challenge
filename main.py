import os
import cv2
from utils import (
    load_map_and_metadata,
    extract_walls,
    detect_lines,
    draw_lines,
    extract_wall_contours,
    extract_wall_segments_as_lines,
    connect_intersecting_lines,
    save_image,
    connect_line_endpoints,
    extract_outer_wall_outline
)
from configs import ROOM_CONFIGS
import numpy as np

# Colors
BG_COLOR = (49, 40, 32)     # #212831 background
LINE_COLOR = (96, 93, 91)   # #5B5D60 line color

DATA_DIR = "./data"
OUTPUT_DIR = "./data/outputs"

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pgm"):
        base = os.path.splitext(filename)[0]
        pgm_path = os.path.join(DATA_DIR, filename)
        yaml_path = os.path.join(DATA_DIR, f"room1.yaml")
        
        # Get room-specific parameters from the configuration
        params = ROOM_CONFIGS.get(base, ROOM_CONFIGS[base])

        print(f"Processing {filename}...")

        # === 1. Load image and metadata
        image, metadata = load_map_and_metadata(pgm_path, yaml_path)

        # === 2. Wall extraction (binary and edge)
        cleaned_binary, _ = extract_walls(image)

        # === 3. Extract raw and filtered contours
        raw_contours = extract_wall_contours(cleaned_binary, min_area=0)
        filtered_contours = extract_wall_contours(cleaned_binary, min_area=params["min_area"])

        # === 3.5 Recompute edges on filtered contours
        edges = cv2.Canny(filtered_contours, 50, 150)

        # === 4. Detect lines (Hough)
        lines = detect_lines(edges, min_line_length=params["min_line_length"], max_line_gap=params["max_line_gap"])
        # Draw raw Hough lines and connected endpoints
        line_image = connect_line_endpoints(lines, edges.shape, max_distance=params["max_distance"])

        # Save the raw version before connecting intersections
        line_image_raw = line_image.copy()

        # Connect extended intersections
        line_image = connect_intersecting_lines(lines, line_image, 
                                                min_intersection_dist=params["min_intersection_dist"],
                                                extend_length=params["extend_length"])

        # === 5. Polygonal wall simplification as straight lines
        polygonal_walls = extract_wall_segments_as_lines(filtered_contours)

        # === 6. Combine everything (polygon walls + straight lines)
        # Ensure both images are grayscale and same size
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if line_image.ndim == 3 else line_image
        poly_gray = cv2.cvtColor(polygonal_walls, cv2.COLOR_BGR2GRAY) if polygonal_walls.ndim == 3 else polygonal_walls

        if line_gray.shape != poly_gray.shape:
            line_gray = cv2.resize(line_gray, (poly_gray.shape[1], poly_gray.shape[0]))

        # Combine
        combined = cv2.bitwise_or(poly_gray, line_gray)

        # === 7. Save intermediate outputs
        room_output_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(room_output_dir, exist_ok=True)

        save_image(image, os.path.join(room_output_dir, "1_original.png"))
        save_image(cleaned_binary, os.path.join(room_output_dir, "2_binary.png"))
        save_image(edges, os.path.join(room_output_dir, "3_edges.png"))
        save_image(raw_contours, os.path.join(room_output_dir, "4_contours.png"))
        save_image(filtered_contours, os.path.join(room_output_dir, "5_contours_filtered.png"))
        save_image(polygonal_walls, os.path.join(room_output_dir, "6_polygonal_walls.png"))
        save_image(line_image_raw, os.path.join(room_output_dir, "7a_lines_raw.png"))
        save_image(line_image, os.path.join(room_output_dir, "7b_lines_connected.png"))

        _, combined = cv2.threshold(combined, 1, 255, cv2.THRESH_BINARY)
        save_image(combined, os.path.join(room_output_dir, "8_combined_walls.png"))

        # === 8. Style and visualize final result
        styled = np.full((*combined.shape, 3), BG_COLOR, dtype=np.uint8)

        # Draw outer wall outline
        outer_wall = extract_outer_wall_outline(combined, params["morph_param"], params["morph_iter"])
        contours, _ = cv2.findContours(outer_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(styled, contours, -1, LINE_COLOR, thickness=3)

        # Save styled output and binary export
        save_image(styled, os.path.join(room_output_dir, "9_output.png"))

        print(f"Saved processed images in {room_output_dir}")

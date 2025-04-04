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
)
from configs import ROOM_CONFIGS

DATA_DIR = "./data"

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pgm"):
        base = os.path.splitext(filename)[0]
        pgm_path = os.path.join(DATA_DIR, filename)
        yaml_path = os.path.join(DATA_DIR, f"{base}.yaml")
        
        # Get room-specific parameters from the configuration
        params = ROOM_CONFIGS.get(base, ROOM_CONFIGS['room1'])

        print(f"Processing {filename}...")

        # === 1. Load image and metadata
        image, metadata = load_map_and_metadata(pgm_path, yaml_path)

        # === 2. Wall extraction (binary and edge)
        cleaned_binary, edges = extract_walls(image)

        # === 3. Extract raw and filtered contours
        raw_contours = extract_wall_contours(cleaned_binary, min_area=0)  # all
        filtered_contours = extract_wall_contours(cleaned_binary, min_area=params["min_area"])  # walls only

        # === 4. Detect lines (Hough)
        lines = detect_lines(edges, min_line_length=params["min_line_length"], max_line_gap=params["max_line_gap"])
        line_image = connect_intersecting_lines(lines, edges.shape, max_distance=params["max_distance"])

        # === 5. Polygonal wall simplification as straight lines
        polygonal_walls = extract_wall_segments_as_lines(filtered_contours)
        polygonal_walls = connect_intersecting_lines(polygonal_walls, min_intersection_dist=params["min_intersection_dist"], extend_length=params["extend_length"])

        # === 6. Combine everything (polygon walls + straight lines)
        # Ensure both images are grayscale and same size
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if line_image.ndim == 3 else line_image
        poly_gray = cv2.cvtColor(polygonal_walls, cv2.COLOR_BGR2GRAY) if polygonal_walls.ndim == 3 else polygonal_walls

        if line_gray.shape != poly_gray.shape:
            line_gray = cv2.resize(line_gray, (poly_gray.shape[1], poly_gray.shape[0]))

        # Combine safely
        combined = cv2.bitwise_or(poly_gray, line_gray)

        # === 7. Save all outputs
        room_output_dir = os.path.join(DATA_DIR, base)
        os.makedirs(room_output_dir, exist_ok=True)

        save_image(image, os.path.join(room_output_dir, "1_original.png"))
        save_image(cleaned_binary, os.path.join(room_output_dir, "2_binary.png"))
        save_image(edges, os.path.join(room_output_dir, "3_edges.png"))
        save_image(raw_contours, os.path.join(room_output_dir, "4_contours.png"))
        save_image(filtered_contours, os.path.join(room_output_dir, "5_contours_filtered.png"))
        save_image(polygonal_walls, os.path.join(room_output_dir, "6_polygonal_walls.png"))
        save_image(line_image, os.path.join(room_output_dir, "7_lines.png"))
        save_image(combined, os.path.join(room_output_dir, "8_combined_walls.png"))

        print(f"Saved processed images in {room_output_dir}")

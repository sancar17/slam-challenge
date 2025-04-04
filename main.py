import os
import cv2
from utils import (
    load_map_and_metadata,
    extract_walls,
    detect_lines,
    draw_lines,
    extract_wall_contours,
    polygonize_contours,
    save_image,
)

DATA_DIR = "./data"

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pgm"):
        base = os.path.splitext(filename)[0]
        pgm_path = os.path.join(DATA_DIR, filename)
        yaml_path = os.path.join(DATA_DIR, f"room1.yaml")

        print(f"Processing {filename}...")

        # Create output folder
        room_output_dir = os.path.join(DATA_DIR, base)
        os.makedirs(room_output_dir, exist_ok=True)

        # === 1. Load image and metadata
        image, metadata = load_map_and_metadata(pgm_path, yaml_path)

        # === 2. Wall extraction (binary and edge)
        cleaned_binary, edges = extract_walls(image)

        # === 3. Extract raw and filtered contours
        raw_contours = extract_wall_contours(cleaned_binary, min_area=0)          # all
        filtered_contours = extract_wall_contours(cleaned_binary, min_area=500)   # walls only

        # === 4. Straight line detection (Hough)
        lines = detect_lines(edges)
        line_image = draw_lines(edges.shape, lines)

        # === 5. Polygonal wall simplification
        polygonal_walls = polygonize_contours(filtered_contours)

        # === 6. Combine everything (polygon walls + straight lines)
        combined = cv2.bitwise_or(polygonal_walls, cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY))

        # === 7. Save all outputs
        save_image(image, os.path.join(room_output_dir, "1_original.png"))
        save_image(cleaned_binary, os.path.join(room_output_dir, "2_binary.png"))
        save_image(edges, os.path.join(room_output_dir, "3_edges.png"))
        save_image(raw_contours, os.path.join(room_output_dir, "4_contours.png"))
        save_image(filtered_contours, os.path.join(room_output_dir, "5_contours_filtered.png"))
        save_image(polygonal_walls, os.path.join(room_output_dir, "6_polygonal_walls.png"))
        save_image(line_image, os.path.join(room_output_dir, "7_lines.png"))
        save_image(combined, os.path.join(room_output_dir, "8_combined_walls.png"))

        print(f"Saved processed images in {room_output_dir}")

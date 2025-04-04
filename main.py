import os
from utils import load_map_and_metadata, extract_walls, save_image

DATA_DIR = "./data"
OUTPUT_DIR = "./data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pgm"):
        base = os.path.splitext(filename)[0]
        pgm_path = os.path.join(DATA_DIR, filename)
        yaml_path = os.path.join(DATA_DIR, f"room1.yaml")

        print(f"Processing {filename}...")

        image, metadata = load_map_and_metadata(pgm_path, yaml_path)
        cleaned, edges = extract_walls(image)

        # Save outputs
        save_image(image, os.path.join(OUTPUT_DIR, f"{base}_original.png"))
        save_image(cleaned, os.path.join(OUTPUT_DIR, f"{base}_binary.png"))
        save_image(edges, os.path.join(OUTPUT_DIR, f"{base}_edges.png"))

        print(f"Saved processed images for {base}")

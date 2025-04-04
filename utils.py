import yaml
import cv2

#helper functinon to load files
def load_map_and_metadata(pgm_path, yaml_path):
    # Load PGM image
    image = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

    # Load YAML metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    return image, metadata

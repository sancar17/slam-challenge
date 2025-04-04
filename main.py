import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils import load_map_and_metadata

if __name__ == "__main__":
    pgm_path = "data/room1.pgm"
    yaml_path = "data/room1.yaml"

    image, metadata = load_map_and_metadata(pgm_path, yaml_path)

    print("Map resolution:", metadata['resolution'])
    print("Map origin:", metadata['origin'])

    plt.imshow(image, cmap='gray')
    plt.title("Room 1 Map")
    plt.axis("off")
    plt.show()

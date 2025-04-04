# SLAM Post-Processing and Floor Plan Rendering

This project performs post-processing on SLAM-generated `.pgm` occupancy grid maps and generates cleaned, stylized floor plans by filtering noise, connecting structural lines, and detecting walls.

---

## Project Structure

```bash
. 
├── data/ 
│ ├── room1.pgm 
│ ├── room1.yaml 
│ ├── ... # Input rooms 
│ └── outputs/ # Folder containing the visuals of each processing step and final outcome of 3 rooms
├── configs.py # Room-specific processing parameters 
├── main.py # Main processing pipeline 
├── utils.py # Utility functions 
├── requirements.txt # All dependencies 
└── README.md
```

## How to Run

### 1. Create environment

```bash
conda create -n slam-challenge python=3.10
conda activate slam-challenge
pip install -r requirements.txt
```

### 2. Add your .pgm and .yaml files under data/

Name them as room1.pgm, room1.yaml, etc.

### 3. Run the pipeline

```bash
python main.py
```

### 4. Output
Processed outputs for each room will be saved to ./data/outputs/<room_name>/ including:

- Cleaned binary masks
- Extracted edges and contours
- Simplified polygonal walls
- Connected Hough lines
- Combined wall structure
- Refined outer wall outline
- Final styled visualization

## Code Overview

### main.py

This is the main pipeline:

Loads SLAM map and YAML metadata
Extracts wall structures (binary + edges)
Filters and simplifies contours
Detects Hough lines and connects endpoints
Combines straight and polygonal wall approximations
Extracts outer wall outline for refinement
Stylizes the result using specified background and line color
Saves all outputs

### configs.py

Contains room-specific hyperparameters such as:

- min_area: filter small noise regions
- epsilon: polygon simplification precision
- min_line_length / max_line_gap: Hough line parameters
- max_distance: max gap to connect endpoints
- morph_param, morph_iter: morphological closing for outer wall smoothing

### utils.py

Contains core image processing functions like:

- extract_walls, detect_lines, extract_wall_contours
- extract_wall_segments_as_lines: polygonal wall simplification
- connect_line_endpoints, connect_intersecting_lines
- extract_outer_wall_outline: final smoothing and outer boundary simplification

## Future Work

Possible future improvements:

- Door detection from structural wall gaps or shape priors
- Interior wall classification vs outer wall segmentation
- Furniture detection from isolated blobs or layout rules
- Learning-based refinement using annotated floor plan datasets
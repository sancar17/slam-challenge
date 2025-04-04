#min_area - minimum wall size to filter
#epsilon - polygon simplification (Approximation precision for contour simplification)
#min_line_length, max_line_gap – Hough line detection
#max_distance – max distance to connect broken wall line endpoints
#min_intersection_dist, extend_length – intersection-based wall connection

ROOM_CONFIGS = {
    "room1": {
        "min_area": 1000,
        "max_distance" : 100,
        "epsilon": 0.03,
        "extend_length": 50,
        "min_intersection_dist":10,
        "min_line_length":60,
        "max_line_gap":150,
        "morph_param":3,
        "morph_iter":95
    },
    "room2": {
        "min_area": 200,
        "max_distance" : 100,
        "epsilon": 0.03,
        "extend_length": 50,
        "min_intersection_dist":20,
        "min_line_length":10,
        "max_line_gap":40,
        "morph_param":3,
        "morph_iter":40
    },
    "room3": {
        "min_area": 500,
        "max_distance" : 100,
        "epsilon": 1,
        "extend_length": 200,
        "min_intersection_dist":100,
        "min_line_length":10,
        "max_line_gap":40,
        "morph_param":2,
        "morph_iter":30
    },
}

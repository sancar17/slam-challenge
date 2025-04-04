#min_area - minimum wall size to filter
#epsilon - polygon simplification (Approximation precision for contour simplification)
#min_line_length, max_line_gap – Hough line detection
#max_distance – max distance to connect broken wall line endpoints
#min_intersection_dist, extend_length – intersection-based wall connection

ROOM_CONFIGS = {
    "room1": {
        "min_area": 200,
        "max_distance" : 2,
        "epsilon": 0.015,
        "connect_gap": 15,
        "extend_length": 25,
        "min_intersection_dist":20,
        "min_line_length":10,
        "max_line_gap":100

    },
    "room2": {
        "min_area": 200,
        "max_distance" : 2,
        "epsilon": 0.015,
        "connect_gap": 15,
        "extend_length": 25,
        "min_intersection_dist":20,
        "min_line_length":10,
        "max_line_gap":100
    },
    "room3": {
        "min_area": 200,
        "max_distance" : 2,
        "epsilon": 0.015,
        "connect_gap": 15,
        "extend_length": 25,
        "min_intersection_dist":20,
        "min_line_length":10,
        "max_line_gap":100
    },
}

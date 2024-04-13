def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    # p1 and p2 contain x & y of center and keypoint
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index*2], keypoints[keypoint_index*2 +1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance= distance
            key_point_ind = keypoint_index
    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
def get_center_of_bbox(ball_box):
    return (int((ball_box[0]+ball_box[2])/2), int((ball_box[1] + ball_box[3])/2))
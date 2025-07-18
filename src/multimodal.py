import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import cv2

# === Configuration ===
SEG_MODEL_PATH = 'yolov8m-seg.pt'
POSE_MODEL_PATH = 'yolov8m-pose.pt'
KEYPOINT_CONF_THRESH = 0.3     # Minimum wrist keypoint confidence
DIST_THRESH = 20.0             # Max pixel distance from wrist to mask contour
SEG_CONF_THRESH = 0.25         # Minimum segmentation confidence
MATCH_BOOST = 0.5              # Confidence boost for mask–wrist matches

# === Load Models ===
seg_model = YOLO(SEG_MODEL_PATH)
pose_model = YOLO(POSE_MODEL_PATH)

def infer_multimodal(image):
    """
    Runs segmentation and pose, then fuses wrist keypoints with glove masks.
    Returns a list of final detections with updated confidences and masks.
    """
    seg_results = seg_model(image)[0]
    pose_results = pose_model(image)[0]

    # 2. Extract high-confidence wrist keypoints
    wrist_pts = []
    # COCO wrist indices: 9 = left_wrist, 10 = right_wrist
    for person_idx in range(len(pose_results.keypoints.xy)):
        for w_idx in (9, 10):
            x, y = pose_results.keypoints.xy[person_idx][w_idx]
            c = pose_results.keypoints.conf[person_idx][w_idx]
            if c >= KEYPOINT_CONF_THRESH:
                wrist_pts.append((x, y, c))

    # 3. Build shapely polygons for segmentation masks
    mask_polygons = []
    for mask in seg_results.masks.data:
        mask_polygons.append(Polygon(mask))

    # 4. Match wrists to masks by distance
    matched_mask_indices = set()
    for i, poly in enumerate(mask_polygons):
        for (x, y, _) in wrist_pts:
            if Point(x, y).distance(poly) < DIST_THRESH:
                matched_mask_indices.add(i)
                break

    # 5. Adjust confidences and collect final detections
    final_boxes = []
    for i, det in enumerate(seg_results.boxes):
        conf = float(det.conf)
        if conf >= SEG_CONF_THRESH or i in matched_mask_indices:
            new_conf = max(conf, MATCH_BOOST) if i in matched_mask_indices else conf
            det.conf = new_conf
            det.mask = seg_results.masks.data[i]
            final_boxes.append(det)

    return final_boxes

def visualize_and_save(image, detections, output_path):
    """
    Draws boxes and masks on the image and saves to output_path.
    """
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf_text = f'{det.conf:.2f}'
        cv2.putText(img, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if hasattr(det, 'mask'):
            pts = np.array(det.mask, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imwrite(output_path, img)
    print(f'Results saved to {output_path}')

if __name__ == '__main__':
    # --- Hardcoded image and output path for testing ---
    TEST_IMAGE_PATH = 'test.jpg'       # Place your input image file here
    OUTPUT_IMAGE_PATH = 'output.jpg'   # Output path

    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{TEST_IMAGE_PATH}'")

    detections = infer_multimodal(img)
    visualize_and_save(img, detections, OUTPUT_IMAGE_PATH)

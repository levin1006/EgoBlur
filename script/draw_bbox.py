#!/usr/bin/env python3
# Draw bounding boxes on image

import sys
import cv2
from demo_ego_blur import *

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: draw_bbox.py <model_path> <input_image> <output_image>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_image = sys.argv[2]
    output_image = sys.argv[3]

    # Load model
    device = get_device()
    face_detector = torch.jit.load(model_path, map_location="cpu").to(device)
    face_detector.eval()

    # Read and process image
    bgr_image = read_image(input_image)
    image_tensor = get_image_tensor(bgr_image)
    image_tensor = image_tensor.float()  # Convert to float
    detections = get_detections(face_detector, image_tensor, 0.5, 0.3)

    # Draw bounding boxes
    vis_image = bgr_image.copy()
    for box in detections:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite(output_image, vis_image)
    print(f"Detected {len(detections)} faces")

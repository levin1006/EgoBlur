#!/usr/bin/env python3
# Batch process images with EgoBlur
# Loads model once and processes images sequentially
# Supports separate detection of faces and license plates

import argparse
import os
from pathlib import Path
import cv2
import torch
import numpy as np
import torchvision
from demo_ego_blur import (
    get_device,
    read_image,
    get_image_tensor,
    visualize,
)


def get_detections_with_labels(
    detector,
    image_tensor: torch.Tensor,
    score_threshold: float,
    nms_threshold: float,
    detect_faces: bool,
    detect_lp: bool,
):
    """Get detections with label filtering.

    Args:
        detector: Detection model
        image_tensor: Input image tensor
        score_threshold: Confidence threshold
        nms_threshold: NMS threshold
        detect_faces: Whether to detect faces (label 0)
        detect_lp: Whether to detect license plates (label 1)

    Returns:
        Tuple of (face_boxes, lp_boxes) as lists
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, labels, scores, _ = detections

    # Apply NMS
    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_threshold)
    boxes = boxes[nms_keep_idx]
    labels = labels[nms_keep_idx]
    scores = scores[nms_keep_idx]

    # Apply score threshold
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()
    score_keep_idx = np.where(scores > score_threshold)[0]

    boxes = boxes[score_keep_idx]
    labels = labels[score_keep_idx]

    # Separate by label
    face_boxes = []
    lp_boxes = []

    for box, label in zip(boxes, labels):
        if label == 0 and detect_faces:
            face_boxes.append(box.tolist())
        elif label == 1 and detect_lp:
            lp_boxes.append(box.tolist())

    return face_boxes, lp_boxes


def process_image(
    img_path: str,
    detector,
    threshold: float,
    nms_threshold: float,
    scale_factor: float,
    blur_dir: str,
    bbox_dir: str,
    save_bbox: bool,
    detect_faces: bool,
    detect_lp: bool,
):
    """Process single image and save results.

    Args:
        img_path: Path to input image
        detector: Detection model
        threshold: Detection confidence threshold
        nms_threshold: NMS threshold
        scale_factor: Scale factor for detections
        blur_dir: Output directory for blur images
        bbox_dir: Output directory for bbox images
        save_bbox: Whether to save bbox visualization
        detect_faces: Whether to detect faces
        detect_lp: Whether to detect license plates

    Returns:
        Tuple of (num_faces, num_lp)
    """
    # Read image
    bgr_image = read_image(img_path)
    if bgr_image is None:
        return (-1, -1)

    # Get detections
    image_tensor = get_image_tensor(bgr_image).float()
    face_boxes, lp_boxes = get_detections_with_labels(
        detector, image_tensor, threshold, nms_threshold, detect_faces, detect_lp
    )

    # Generate output filename with detection counts
    base_name = Path(img_path).stem
    num_faces = len(face_boxes)
    num_lp = len(lp_boxes)
    output_filename = f"{base_name}_{num_faces}f_{num_lp}lp.jpg"

    # Combine all detections for blurring
    all_detections = face_boxes + lp_boxes

    # Save blur image
    if all_detections:
        blur_image = visualize(bgr_image.copy(), all_detections, scale_factor)
    else:
        blur_image = bgr_image.copy()

    os.makedirs(blur_dir, exist_ok=True)
    blur_output = os.path.join(blur_dir, output_filename)
    cv2.imwrite(blur_output, blur_image)

    # Save bbox image if requested
    if save_bbox:
        bbox_image = bgr_image.copy()

        # Draw faces in green
        for box in face_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw license plates in blue
        for box in lp_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        os.makedirs(bbox_dir, exist_ok=True)
        bbox_output = os.path.join(bbox_dir, output_filename)
        cv2.imwrite(bbox_output, bbox_image)

    return (num_faces, num_lp)


def process_camera_folder(
    cam_folder: Path,
    cam_name: str,
    detector,
    output_base: str,
    threshold: float,
    nms_threshold: float,
    scale_factor: float,
    max_images: int,
    sample_interval: int,
    save_bbox: bool,
    detect_faces: bool,
    detect_lp: bool,
) -> int:
    """Process all images in a camera folder.

    Args:
        cam_folder: Path to camera folder
        cam_name: Camera folder name
        face_detector: Face detection model
        output_base: Base output directory
        face_threshold: Detection threshold
        nms_threshold: NMS threshold
        scale_factor: Detection scale factor
        max_images: Maximum images to process
        sample_interval: Sample every Nth image
        save_bbox: Whether to save bbox visualization

    Returns:
        Number of images processed
    """
    # Get image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = sorted([
        f for f in cam_folder.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    # Apply sampling and limit
    sampled_images = all_images[::sample_interval]
    if max_images and max_images > 0:
        image_files = sampled_images[:max_images]
    else:
        image_files = sampled_images

    if not image_files:
        print(f"  No images found")
        return 0

    # Setup output directories
    blur_dir = os.path.join(output_base, "blur", cam_name)
    bbox_dir = os.path.join(output_base, "bbox", cam_name)

    # Process images
    total = len(image_files)
    processed = 0

    for i, img_path in enumerate(image_files, 1):
        img_name = img_path.name
        print(f"  [{i}/{total}] {img_name}", end=" ")

        num_faces, num_lp = process_image(
            str(img_path),
            detector,
            threshold,
            nms_threshold,
            scale_factor,
            blur_dir,
            bbox_dir,
            save_bbox,
            detect_faces,
            detect_lp,
        )

        if num_faces >= 0:
            print(f"({num_faces}f, {num_lp}lp) OK")
            processed += 1
        else:
            print("FAILED")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images with EgoBlur (model loaded once)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to face detection model",
    )
    parser.add_argument(
        "--input_base",
        type=str,
        required=True,
        help="Base directory containing camera folders",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Output base directory",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum images per camera (default: all)",
    )
    parser.add_argument(
        "--face_threshold",
        type=float,
        default=0.5,
        help="Face detection threshold",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.3,
        help="NMS threshold",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="Scale factor for detections",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sample every Nth image (1=all)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save bounding box visualization",
    )
    parser.add_argument(
        "--no-detect-faces",
        action="store_false",
        dest="detect_faces",
        help="Disable face detection (label 0)",
    )
    parser.add_argument(
        "--no-detect-lp",
        action="store_false",
        dest="detect_lp",
        help="Disable license plate detection (label 1)",
    )

    args = parser.parse_args()

    # Load model once
    print(f"Loading model: {args.model_path}")
    device = get_device()
    print(f"Using device: {device}")

    face_detector = torch.jit.load(args.model_path, map_location="cpu").to(device)
    face_detector.eval()

    # Get camera folders
    camera_folders = sorted([
        d for d in Path(args.input_base).glob("cam_*")
        if d.is_dir()
    ])

    if not camera_folders:
        print(f"No camera folders found in {args.input_base}")
        return

    print(f"\nProcessing {len(camera_folders)} camera folders...\n")

    # Process each camera folder
    total_processed = 0
    for cam_folder in camera_folders:
        cam_name = cam_folder.name
        print(f"Processing {cam_name}...")

        count = process_camera_folder(
            cam_folder,
            cam_name,
            face_detector,
            args.output_base,
            args.face_threshold,
            args.nms_threshold,
            args.scale_factor,
            args.max_images,
            args.sample_interval,
            args.visualize,
            args.detect_faces,
            args.detect_lp,
        )

        total_processed += count
        print(f"Completed {cam_name}: {count} images\n")

    # Summary
    print(f"\nAll processing complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Output structure:")
    print(f"  {args.output_base}/blur/{{cam_name}}/{{image}}_{{count}}.jpg")
    if args.visualize:
        print(f"  {args.output_base}/bbox/{{cam_name}}/{{image}}_{{count}}.jpg")


if __name__ == "__main__":
    main()

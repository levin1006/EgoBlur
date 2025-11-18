#!/usr/bin/env python3
# Check what labels the model outputs

import sys
import torch
from demo_ego_blur import get_device, read_image, get_image_tensor

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: check_model_labels.py <model_path> <test_image>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Load model
    device = get_device()
    print(f"Loading model: {model_path}")
    model = torch.jit.load(model_path, map_location="cpu").to(device)
    model.eval()

    # Load image
    print(f"Loading image: {image_path}")
    bgr_image = read_image(image_path)
    image_tensor = get_image_tensor(bgr_image).float()

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        detections = model(image_tensor)

    boxes, labels, scores, dims = detections

    print(f"\nDetections found: {len(boxes)}")
    print(f"\nBoxes shape: {boxes.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Dims shape: {dims.shape}")

    # Show labels
    print(f"\nUnique labels: {torch.unique(labels).cpu().tolist()}")
    print(f"\nLabel distribution:")
    for label in torch.unique(labels):
        count = (labels == label).sum().item()
        print(f"  Label {label.item()}: {count} detections")

    # Show top 10 detections
    print(f"\nTop 10 detections:")
    top_indices = torch.argsort(scores, descending=True)[:10]
    for i, idx in enumerate(top_indices, 1):
        label = labels[idx].item()
        score = scores[idx].item()
        box = boxes[idx].cpu().tolist()
        print(f"  {i}. Label: {label}, Score: {score:.3f}, Box: {box}")

    print("\nLabel meaning (guess):")
    print("  0 = face")
    print("  1 = license plate")

#!/usr/bin/env python3
# Create video from multi-camera visualization outputs
# Combines 5 camera views into single frame: bbox and blur side-by-side

import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import subprocess


def get_image_groups(output_base: str):
    """Group images by frame timestamp across cameras.

    Returns:
        dict: {timestamp: {cam_name: {bbox: path, blur: path}}}
    """
    import re

    bbox_dir = Path(output_base) / "bbox"
    blur_dir = Path(output_base) / "blur"

    # Expected camera names
    cameras = [
        "cam_front_tf_left",
        "cam_front_left",
        "cam_front_right",
        "cam_rear_left",
        "cam_rear_right"
    ]

    # Group by timestamp (extracted from filename)
    frame_groups = defaultdict(lambda: defaultdict(dict))

    # Pattern: {prefix}_{CAM_CODE}_10_{TIMESTAMP}_{Xf_Ylp}.jpg
    # Extract timestamp which appears after "_10_"
    timestamp_pattern = re.compile(r'_10_(\d+)_\d+f_\d+lp')

    for cam_name in cameras:
        bbox_cam_dir = bbox_dir / cam_name
        blur_cam_dir = blur_dir / cam_name

        if not bbox_cam_dir.exists() or not blur_cam_dir.exists():
            continue

        # Get all bbox images for this camera
        for bbox_file in sorted(bbox_cam_dir.glob("*.jpg")):
            filename = bbox_file.stem

            # Extract timestamp
            match = timestamp_pattern.search(filename)
            if match:
                timestamp = match.group(1)
            else:
                # Fallback: try to extract timestamp from end
                parts = filename.rsplit("_", 2)
                if len(parts) >= 3 and parts[-1].endswith("lp") and parts[-2].endswith("f"):
                    # Try to find timestamp in parts[0]
                    timestamp_parts = parts[0].split("_")
                    if timestamp_parts:
                        timestamp = timestamp_parts[-1]
                    else:
                        continue
                else:
                    continue

            # Find corresponding blur image
            blur_file = blur_cam_dir / bbox_file.name

            if blur_file.exists():
                frame_groups[timestamp][cam_name]["bbox"] = str(bbox_file)
                frame_groups[timestamp][cam_name]["blur"] = str(blur_file)

    return frame_groups


def create_composite_frame(
    frame_data: dict,
    cameras: list,
    scale: float,
    output_size: tuple = None
) -> np.ndarray:
    """Create composite frame from 5 camera views.

    Layout:
        [    front_tf_left (full)     ]
        [front_left][front_right]
        [rear_left ][rear_right ]

    Args:
        frame_data: {cam_name: {bbox: path, blur: path}}
        cameras: List of camera names in order
        scale: Scale factor for output
        output_size: Optional (width, height) to resize final output

    Returns:
        Composite frame with bbox on left, blur on right
    """
    # Camera layout positions
    # 0: front_tf_left (top, full width)
    # 1: front_left (middle left)
    # 2: front_right (middle right)
    # 3: rear_left (bottom left)
    # 4: rear_right (bottom right)

    # Load all images
    bbox_images = {}
    blur_images = {}

    for cam in cameras:
        if cam not in frame_data:
            return None
        if "bbox" not in frame_data[cam] or "blur" not in frame_data[cam]:
            return None

        bbox_img = cv2.imread(frame_data[cam]["bbox"])
        blur_img = cv2.imread(frame_data[cam]["blur"])

        if bbox_img is None or blur_img is None:
            return None

        bbox_images[cam] = bbox_img
        blur_images[cam] = blur_img

    # Get reference size from front_tf_left
    ref_img = bbox_images[cameras[0]]
    h, w = ref_img.shape[:2]

    # Apply scale
    w_scaled = int(w * scale)
    h_scaled = int(h * scale)
    half_w = w_scaled // 2
    half_h = h_scaled // 2

    # Create canvas for single composite (5 cameras)
    # Height: h_scaled (top) + half_h (middle) + half_h (bottom) = 2*h_scaled
    canvas_h = 2 * h_scaled
    canvas_w = w_scaled

    def create_single_composite(images):
        """Create composite from either bbox or blur images."""
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Resize and place images
        # Top: front_tf_left (full width)
        img_top = cv2.resize(images[cameras[0]], (w_scaled, h_scaled))
        canvas[0:h_scaled, 0:w_scaled] = img_top

        # Middle left: front_left (half size)
        img_ml = cv2.resize(images[cameras[1]], (half_w, half_h))
        canvas[h_scaled:h_scaled+half_h, 0:half_w] = img_ml

        # Middle right: front_right (half size)
        img_mr = cv2.resize(images[cameras[2]], (half_w, half_h))
        canvas[h_scaled:h_scaled+half_h, half_w:w_scaled] = img_mr

        # Bottom left: rear_left (half size)
        img_bl = cv2.resize(images[cameras[3]], (half_w, half_h))
        canvas[h_scaled+half_h:canvas_h, 0:half_w] = img_bl

        # Bottom right: rear_right (half size)
        img_br = cv2.resize(images[cameras[4]], (half_w, half_h))
        canvas[h_scaled+half_h:canvas_h, half_w:w_scaled] = img_br

        return canvas

    # Create bbox and blur composites
    bbox_composite = create_single_composite(bbox_images)
    blur_composite = create_single_composite(blur_images)

    # Combine side by side: bbox | blur
    final_frame = np.hstack([bbox_composite, blur_composite])

    # Optional resize to specific output size
    if output_size:
        final_frame = cv2.resize(final_frame, output_size)

    return final_frame


def create_video(
    output_base: str,
    output_video: str,
    scale: float = 0.5,
    fps: int = 10,
    output_width: int = None,
):
    """Create video from visualization outputs.

    Args:
        output_base: Base output directory containing bbox/ and blur/
        output_video: Output video path
        scale: Scale factor for composite (default 0.5)
        fps: Video frame rate (default 10)
        output_width: Optional output width (maintains aspect ratio)
    """
    cameras = [
        "cam_front_tf_left",
        "cam_front_left",
        "cam_front_right",
        "cam_rear_left",
        "cam_rear_right"
    ]

    print("Scanning images...")
    frame_groups = get_image_groups(output_base)

    if not frame_groups:
        print("No frames found!")
        return

    # Sort timestamps
    timestamps = sorted(frame_groups.keys())
    print(f"Found {len(timestamps)} frames")

    # Create temporary directory for composite frames
    temp_dir = Path(output_base) / "video_frames"
    temp_dir.mkdir(exist_ok=True)

    print(f"Creating composite frames with scale={scale}...")

    # Determine output size if width is specified
    output_size = None
    first_frame = None

    valid_frames = 0
    for i, timestamp in enumerate(timestamps, 1):
        frame_data = frame_groups[timestamp]

        # Check if all cameras are present
        missing_cameras = [cam for cam in cameras if cam not in frame_data]
        if missing_cameras:
            print(f"  [{i}/{len(timestamps)}] {timestamp}: SKIP (missing {', '.join(missing_cameras)})")
            continue

        # Create composite
        composite = create_composite_frame(frame_data, cameras, scale, output_size)

        if composite is None:
            print(f"  [{i}/{len(timestamps)}] {timestamp}: SKIP (composite failed)")
            continue

        # Calculate output size from first frame
        if first_frame is None and output_width:
            h, w = composite.shape[:2]
            aspect_ratio = h / w
            output_height = int(output_width * aspect_ratio)
            output_size = (output_width, output_height)
            composite = cv2.resize(composite, output_size)
            first_frame = composite

        # Save frame
        frame_path = temp_dir / f"frame_{valid_frames:06d}.jpg"
        cv2.imwrite(str(frame_path), composite)

        valid_frames += 1
        if i % 10 == 0 or i == len(timestamps):
            print(f"  [{i}/{len(timestamps)}] {timestamp}: OK")

    if valid_frames == 0:
        print("No valid frames to create video!")
        return

    print(f"\nCreating video: {output_video}")
    print(f"  Frames: {valid_frames}")
    print(f"  FPS: {fps}")

    # Create video with ffmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(temp_dir / "frame_*.jpg"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n✓ Video created: {output_video}")

        # Get video info
        video_info = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size", "-of", "default=noprint_wrappers=1",
             output_video],
            capture_output=True,
            text=True
        )
        print(f"  {video_info.stdout.strip()}")

        # Clean up temp frames
        print(f"\nCleaning up temporary frames...")
        for frame_file in temp_dir.glob("frame_*.jpg"):
            frame_file.unlink()
        temp_dir.rmdir()

    else:
        print(f"\n✗ FFmpeg error:")
        print(result.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Create video from multi-camera visualization outputs"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base output directory containing bbox/ and blur/ folders"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Output video file path (.mp4)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor for composite (default: 0.5)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frame rate (default: 10)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output video width in pixels (height calculated to maintain aspect ratio)"
    )

    args = parser.parse_args()

    create_video(
        args.output_base,
        args.output_video,
        args.scale,
        args.fps,
        args.width
    )


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EgoBlur is a face and license plate anonymization system from Meta/Project Aria. It has two components:
1. **Python CLI** (`script/demo_ego_blur.py`) - Process images and videos
2. **C++ VRS Tool** (`tools/vrs_mutation/`) - Process VRS format files (Project Aria recordings)

Models must be downloaded from https://www.projectaria.com/tools/egoblur/

## Build Commands

### Python Setup
```bash
conda env create --file=environment.yaml
conda activate ego_blur
```

### C++ VRS Tool Build
Requires CMake 3.28+, LibTorch 2.1 (CUDA 12.1), and TorchVision 0.16.0 built from source.

```bash
cd tools/vrs_mutation
mkdir build && cd build
cmake .. -DTorch_DIR=~/libtorch/share/cmake/Torch -DTorchVision_DIR=~/repos/vision/cmake
make -j ego_blur_vrs_mutation
```

See `tools/README.md` for full dependency installation instructions.

## Run Commands

### Python Demo
```bash
# Image processing
python script/demo_ego_blur.py \
  --face_model_path /path/to/ego_blur_face.jit \
  --input_image_path demo_assets/test_image.jpg \
  --output_image_path output.jpg

# Video processing
python script/demo_ego_blur.py \
  --face_model_path /path/to/ego_blur_face.jit \
  --lp_model_path /path/to/ego_blur_lp.jit \
  --input_video_path input.mp4 \
  --output_video_path output.mp4
```

### C++ VRS Tool
```bash
./ego_blur_vrs_mutation \
  --in input.vrs \
  --out output.vrs \
  -f ~/models/ego_blur_face.jit \
  -l ~/models/ego_blur_lp.jit \
  --use-gpu
```

## Testing

No automated test suite. Manual testing only:
- Use `demo_assets/test_image.jpg` and `demo_assets/test_video.mp4` for Python validation
- Built-in input validation raises `ValueError` for invalid arguments

## Architecture

### Python Component (`script/demo_ego_blur.py`)
- Single 512-line script with comprehensive argument validation
- Pipeline: Load TorchScript models -> Detect faces/LPs -> Apply NMS -> Blur with elliptical mask
- Auto-detects GPU with CPU fallback
- Key functions: `validate_inputs()`, `get_detections()`, `visualize()`, `visualize_video()`

### C++ Component (`tools/vrs_mutation/`)
- Header-only implementation in `EgoBlurImageMutator.h`
- Inherits from `UserDefinedImageMutator` for VRS frame processing
- Explicit GPU memory management with `CUDACachingAllocator::emptyCache()` per frame
- Filters Project Aria streams by ID: "214" (RGB), "1201" (grayscale)

### Key Patterns
- **Early validation**: Both components validate all inputs before processing
- **Device caching**: Python uses `@lru_cache` for device selection
- **Memory management**: C++ explicitly resets tensors and clears CUDA cache to avoid OOM on long videos
- **Detection flow**: Model inference -> confidence thresholding -> NMS -> scale boxes -> apply blur

## CLI Parameters (defaults)
- `--face_model_score_threshold`: 0.1
- `--lp_model_score_threshold`: 0.1
- `--nms_iou_threshold`: 0.3
- `--scale_factor_detections`: 1.0 (Python), 1.15 (C++)
- `--output_video_fps`: 30

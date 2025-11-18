#!/bin/bash

# EgoBlur face and license plate detection test script
# Loads model once and processes images sequentially (model doesn't support batch inference)
# Usage:
#   ./run_ego_blur_test.sh                           # default settings
#   SAMPLE_INTERVAL=10 ./run_ego_blur_test.sh       # every 10th image
#   MAX_IMAGES=20 ./run_ego_blur_test.sh            # first 20 images only
#   VISUALIZE=false ./run_ego_blur_test.sh          # blur only, no bbox
#   DETECT_FACES=false ./run_ego_blur_test.sh       # license plates only
#   DETECT_LP=false ./run_ego_blur_test.sh          # faces only

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ego_blur

MODEL_PATH="/home/user/Workspace/EgoBlur/models/ego_blur_face.jit"
INPUT_BASE="/mnt/NAS2/30_VISION_DEV/a2z_datasets/Object_Detection/KR/Dongdaemun/250723/AIFW_out/250723_KR_DongDamun_10"
OUTPUT_BASE="$INPUT_BASE/output"

# Options
VISUALIZE="${VISUALIZE:-true}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
MAX_IMAGES="${MAX_IMAGES:-}"
DETECT_FACES="${DETECT_FACES:-true}"
DETECT_LP="${DETECT_LP:-false}"

# Build visualization flag
VIS_FLAG=""
if [ "$VISUALIZE" = "true" ]; then
    VIS_FLAG="--visualize"
fi

# Build max_images flag
MAX_FLAG=""
if [ -n "$MAX_IMAGES" ]; then
    MAX_FLAG="--max_images $MAX_IMAGES"
fi

# Build detection flags
FACES_FLAG=""
if [ "$DETECT_FACES" = "false" ]; then
    FACES_FLAG="--no-detect-faces"
fi

LP_FLAG=""
if [ "$DETECT_LP" = "false" ]; then
    LP_FLAG="--no-detect-lp"
fi

# Run processing (model loaded once, images processed sequentially)
python script/batch_process.py \
    --model_path "$MODEL_PATH" \
    --input_base "$INPUT_BASE" \
    --output_base "$OUTPUT_BASE" \
    $MAX_FLAG \
    --sample_interval "$SAMPLE_INTERVAL" \
    --face_threshold 0.5 \
    $VIS_FLAG \
    $FACES_FLAG \
    $LP_FLAG

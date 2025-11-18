#!/bin/bash

# Create visualization video from batch processing outputs
# Combines 5 camera views: bbox and blur side-by-side
# Usage:
#   ./create_visualization_video.sh                           # default settings
#   SCALE=0.3 ./create_visualization_video.sh                # smaller composite
#   FPS=30 ./create_visualization_video.sh                   # higher frame rate
#   WIDTH=1920 ./create_visualization_video.sh               # specific output width

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ego_blur

OUTPUT_BASE="/mnt/NAS2/30_VISION_DEV/a2z_datasets/Object_Detection/KR/Dongdaemun/250723/AIFW_out/250723_KR_DongDamun_10/output"
OUTPUT_VIDEO="$OUTPUT_BASE/visualization.mp4"

# Options
SCALE="${SCALE:-0.5}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-}"

# Build width flag
WIDTH_FLAG=""
if [ -n "$WIDTH" ]; then
    WIDTH_FLAG="--width $WIDTH"
fi

# Run video creation
python script/create_video.py \
    --output_base "$OUTPUT_BASE" \
    --output_video "$OUTPUT_VIDEO" \
    --scale "$SCALE" \
    --fps "$FPS" \
    $WIDTH_FLAG

echo ""
echo "Video saved to: $OUTPUT_VIDEO"

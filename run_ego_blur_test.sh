#!/bin/bash

# EgoBlur face detection test script
# Processes top 10 images from each camera folder

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ego_blur

MODEL_PATH="/home/user/Workspace/EgoBlur/models/ego_blur_face.jit"
INPUT_BASE="/mnt/NAS2/30_VISION_DEV/a2z_datasets/Object_Detection/KR/Dongdaemun/250723/AIFW_out/250723_KR_DongDamun_0"
OUTPUT_BASE="/home/user/Workspace/EgoBlur/output/250723_KR_DongDamun_0"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Get all camera folders
CAMERA_FOLDERS=$(ls -d "$INPUT_BASE"/cam_*)

for cam_folder in $CAMERA_FOLDERS; do
    cam_name=$(basename "$cam_folder")
    echo "Processing $cam_name..."

    # Create output folder for this camera
    output_folder="$OUTPUT_BASE/$cam_name"
    mkdir -p "$output_folder"

    # Get top 10 images (sorted by name)
    images=$(ls "$cam_folder"/*.jpg 2>/dev/null | sort | head -10)

    for img_path in $images; do
        img_name=$(basename "$img_path")
        output_path="$output_folder/$img_name"

        echo "  Processing: $img_name"

        python script/demo_ego_blur.py \
            --face_model_path "$MODEL_PATH" \
            --input_image_path "$img_path" \
            --output_image_path "$output_path"
    done

    echo "Completed $cam_name: $(ls "$output_folder" | wc -l) images processed"
done

echo ""
echo "All processing complete. Output saved to: $OUTPUT_BASE"

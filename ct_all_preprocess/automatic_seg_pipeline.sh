cd
cd FemurSegmentation/

#!/bin/bash

input_directory="/home/rgu/Documents/UK dataset/leg_img"
output_directory="/home/rgu/Documents/UK dataset/results"

for input_file in "$input_directory"/*.nii.gz; do
    base_name=$(basename "$input_file" .nii.gz)

    echo "Processing: $base_name"

    output_prefix="${output_directory}/${base_name}"

    python3 split_image.py --input "$input_file" --output "$output_prefix"

    segment_output_r="${output_prefix}_1.nrrd"
    segment_output_l="${output_prefix}_2.nrrd"
    right_output="${output_directory}/right_seg_${base_name}.nrrd"
    left_output="${output_directory}/left_seg_${base_name}.nrrd"

    python3 run_automated_segmentation.py --input "$segment_output_r" --output "$right_output"
    python3 run_automated_segmentation.py --input "$segment_output_l" --output "$left_output"

    echo "Completed: $base_name"
done




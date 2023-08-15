cd
cd FemurSegmentation/

#!/bin/bash

input_directory="/home/rgu/Documents/UK dataset/seperated_leg_nrrd/"
output_directory="/home/rgu/Documents/UK dataset/results"

for input_file in "$input_directory"/*; do
    base_name=$(basename "$input_file")

    echo "Processing: $base_name"

    output_file="${output_directory}/seg_${base_name}"

    python3 run_automated_segmentation.py --input "$input_file" --output "$output_file"

    echo "Completed: $output_file"
done




#!/bin/bash

data_root="/home/rgu/Documents/"  
training_dataset="Dataset1000_NMDID" 

nnunet_dataset_path="$data_root/UK_dataset/nnUNet_raw/$training_dataset"
nnunet_label_dir="$data_root/UK_dataset/nnUNet_raw/$training_dataset/labelsTr"
nnunet_image_dir="$data_root/UK_dataset/nnUNet_raw/$training_dataset/imagesTr"

check_path_exist() {
    if [ ! -d "$1" ]; then
        echo "Directory '$1' does not exist."
        exit 1
    fi
}

# Uncomment if to check if directories exist
check_path_exist "$nnunet_dataset_path"
check_path_exist "$nnunet_label_dir"
check_path_exist "$nnunet_image_dir"


addtional_label=($(ls "$data_root/UK_dataset/nnUNet_raw/Dataset101_leg/labelsTr"/*.nii.gz))
addtional_image=($(ls "$data_root/UK_dataset/nnUNet_raw/Dataset101_leg/imagesTr"/*.nii.gz))

# Copy additional data to the nnU-Net directories
cp "${addtional_image[@]}" "$nnunet_image_dir"
cp "${addtional_label[@]}" "$nnunet_label_dir"

echo "Additional data copied"

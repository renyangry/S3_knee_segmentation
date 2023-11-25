export nnUNet_raw="/home/rgu/Documents/UK dataset/nnUNet_raw"
export nnUNet_preprocessed="/home/rgu/Documents/UK dataset/nnUNet_raw/preprocessed"
export nnUNet_results="/home/rgu/Documents/UK dataset/nnUNet_raw/results"

# nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres -np 4 #--verify_dataset_integrity
#nnUNetv2_train 725 3d_fullres 4
# nnUNetv2_train 101 3d_fullres all -device 'cuda'
nnUNetv2_predict -i /home/rgu/Documents/UK\ dataset/nnUNet_testing -o /home/rgu/Documents/UK\ dataset/nnUNet_raw/testing_dual_label -d 101 -chk checkpoint_best.pth -c 3d_fullres -f all -device 'cuda'
# nnUNetv2_predict -i /home/rgu/Documents/UK\ dataset/nnUNet_testing -o /home/rgu/Documents/UK\ dataset/nnUNet_raw/testing_ps -d 104 -c 3d_fullres -f 5 


# nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_DiceTopK10 TaskXX_MY_DATASET 4

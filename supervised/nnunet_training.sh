#export nnUNet_raw="/home/rgu/Documents/UK dataset/nnUNet_raw"
#export nnUNet_preprocessed="/home/rgu/Documents/UK dataset/nnUNet_raw/preprocessed"
#export nnUNet_results="/home/rgu/Documents/UK dataset/nnUNet_raw/results"

# step 1
nnUNetv2_plan_and_preprocess -d 1000 -c 3d_fullres -np 4 #--verify_dataset_integrity

# step 2
#nnUNetv2_train 725 3d_fullres 4      # use this for training
# nnUNetv2_train 101 3d_fullres all -device 'cuda'
# nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_DiceTopK10 TaskXX_MY_DATASET 4 # not updated correctly

# step 3
# nnUNetv2_predict -i /home/rgu/Documents/UK\ dataset/nnUNet_testing -o /home/rgu/Documents/UK\ dataset/nnUNet_raw/testing_dual_label -d 101 -chk checkpoint_best.pth -c 3d_fullres -f all -device 'cuda'
# nnUNetv2_predict -i /home/rgu/Documents/UK\ dataset/nnUNet_testing -o /home/rgu/Documents/UK\ dataset/nnUNet_raw/testing_ps -d 104 -c 3d_fullres -f 5 



import os
import nibabel as nib

data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
training_dataset1 = 'Dataset102_femur'
training_dataset2 = 'Dataset103_tibia'
train_label_dir1 = os.path.join(data_root, which_dataset, 'nnUNet_raw', training_dataset1, 'labelsTr')
train_label_dir2 = os.path.join(data_root, which_dataset, 'nnUNet_raw', training_dataset2, 'labelsTr')
final_dir = os.path.join(data_root, which_dataset, 'nnUNet_raw', 'Dataset104_ps', 'labelsTr')
for i in os.listdir(train_label_dir1):
    if i.endswith('.nii.gz'):
        print(i)   
        label = nib.load(os.path.join(train_label_dir1, i))     
        label_data = label.get_fdata()
        label_data[label_data == 0] = 3
        label_new = nib.Nifti1Image(label_data, label.affine, label.header)
        nib.save(label_new, os.path.join(final_dir, i))


for i in os.listdir(train_label_dir2):
    if i.endswith('.nii.gz'):
        print(i)
        label = nib.load(os.path.join(train_label_dir2, i))
        label_data = label.get_fdata()
        label_data[label_data == 0] = 3
        label_new = nib.Nifti1Image(label_data, label.affine, label.header)
        nib.save(label_new, os.path.join(final_dir, i))



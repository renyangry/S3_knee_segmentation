from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Activationsd,
)
from monai.networks.nets import UNet, VNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
import torch
import glob
import time
from ct_all_preprocess.utils import *
import subprocess

data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
network = 'VNet'
test_folder = 'diceloss'
test_images = sorted(glob.glob(os.path.join(data_root, which_dataset, "seperated_leg_img", "*.nii.gz")))
data_dir = os.path.join(data_root, which_dataset, network, test_folder)

test_data = [{"image": image} for image in test_images]

test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-400,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)

test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    [
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict",
                   output_dir=os.path.join(data_dir, 'results'), output_postfix="",
                   resample=False),
    ]
)

device = torch.device("cuda:0")
# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
# ).to(device)
#version 2 - 3D V-Net
model = VNet(spatial_dims=3, in_channels=1, out_channels=1, act='relu', dropout_prob=0.1, dropout_dim=3).to(device)

model.load_state_dict(torch.load(glob.glob(os.path.join(data_dir, "best_3d_unet.pth"))[-1]))
model.eval()

with torch.no_grad():
    for test_data in test_org_loader:
        start = time.time()
        test_inputs = test_data["image"].to(device)
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        end = time.time()
        elapsed_time = end - start
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Testing Elapsed time: {minutes} minutes {seconds} seconds")

directory = os.path.join(data_root, 'UK\ dataset', network, test_folder, 'results')
bash_script = f"""
#!/bin/bash
cd {directory} 

for d in */; do
  mv "$d"/* .
  rm -r "$d"

done
"""

process = subprocess.Popen(bash_script, shell=True, stdout=subprocess.PIPE)
process.wait()
print(process.stdout.read().decode())

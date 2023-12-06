#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 00:36:00 2023

@author: renne
"""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
import time
from glob import glob
import nibabel as nib

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import monai
from monai.data import list_data_collate, decollate_batch, DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandAffined,
    CropForegroundd,
    Spacingd, ScaleIntensityRanged,
)
from monai.visualize import plot_2d_or_3d_image
import numpy as np

from ct_all_preprocess.utils import check_path_exist, check_num_of_pt_4eval

tempdir = tempfile.TemporaryDirectory()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# create a temporary directory and 40 random image, mask pairs
print(f"generating synthetic data to {tempdir} (this may take a while)")

data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
data_dir = os.path.join(data_root, which_dataset, 'UNet')
check_path_exist(data_dir)
all_train_images = sorted(glob(os.path.join(data_root, which_dataset, "seperated_leg_img", "*.nii.gz")))
all_train_labels = (sorted(glob(os.path.join(data_root, which_dataset, "seperated_leg_seg", "left_*.nii.gz")))) + (
    sorted(glob(os.path.join(data_root, which_dataset, "seperated_leg_seg", "right_*.nii.gz"))))
check_num_of_pt_4eval(all_train_labels, all_train_images)

train_images = []
train_labels = []
for idx, gt_file_path in enumerate(all_train_labels):
    gt_file_name = os.path.split(gt_file_path)[-1][0:12]
    cut_index = next(
        (i for i, cut_file_path in enumerate(all_train_images) if
         os.path.split(cut_file_path)[-1][0:12] == gt_file_name), None)
    if cut_index is not None:
        train_labels.append(all_train_labels[idx])
        train_images.append(all_train_images[cut_index])
check_num_of_pt_4eval(train_labels, train_images)

for patient in range(len(train_images)):
    image_name = train_images[patient]
    label_name = train_labels[patient]
    image = nib.load(image_name).get_fdata()
    label = nib.load(label_name).get_fdata()

    for slice in range(image.shape[2]):
        affine_im = nib.load(image_name).affine
        header_im = nib.load(image_name).header
        img = image[:, :, slice]
        gt = label[:, :, slice]

        ct = nib.Nifti1Image(img, affine=affine_im, header=header_im)
        ms = nib.Nifti1Image(gt, affine=affine_im, header=header_im)

        nib.save(ct, os.path.join(tempdir.name, f"img{slice:d}.nii.gz"))
        nib.save(ms, os.path.join(tempdir.name, f"seg{slice:d}.nii.gz"))

images = sorted(glob(os.path.join(tempdir.name, "img*.nii.gz")))
segs = sorted(glob(os.path.join(tempdir.name, "seg*.nii.gz")))
num_vals = np.int32(len(images) * 0.2)
train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:num_vals], segs[:num_vals])]
val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-num_vals:], segs[-num_vals:])]

# define transforms for image and segmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityRanged(
            keys=["img"],
            a_min=-400,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["img", "seg"], source_key="seg"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg", spatial_size=[64, 64], pos=1, neg=1, num_samples=4
        ),
        RandAffined(
            keys=['img', 'seg'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(64, 64),
            rotate_range=(np.pi / 9, np.pi / 9),
            scale_range=(0.25, 0.25)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityRanged(
            keys=["img"],
            a_min=-400,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["img", "seg"], source_key="seg"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)


# create a training data loader
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
)
# create a validation data loader
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

dice_metric = DiceMetric()
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = monai.losses.DiceCELoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
max_epochs = 1000
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epochs_no_improve = 0
patience = 5
epoch_loss_values = []
epoch_val_loss_values = []
metric_values = []
writer = SummaryWriter()

start_time = time.time()
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # epoch_len = len(train_ds) // train_loader.batch_size
    #        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    #        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f}")
    writer.add_scalar("epoch_loss", epoch_loss, epoch + 1)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                #                writer.add_scalar("val_loss", val_loss.item(), epoch_val_len * epoch + val_step)
                roi_size = (64, 64)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_loss = loss_function(val_outputs, val_labels)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                epoch_val_loss += val_loss.item()
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            writer.add_scalar("val_mean_dice", metric, epoch + 1)

            epoch_val_loss /= len(val_ds)
            epoch_val_loss_values.append(epoch_val_loss)
            print(f"epoch {epoch + 1} average val loss: {epoch_val_loss:.4f}")
            writer.add_scalar("epoch_val_loss", epoch_val_loss, epoch + 1)
            # reset the status for next validation round
            dice_metric.reset()

            if epoch_val_loss > epoch_loss:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                # torch.save(model.state_dict(), "best_metric_2d_unet.pth")
                # print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

            if epochs_no_improve == patience:
                print(f"early stopping at current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}")
                torch.save(model.state_dict(), os.path.join(data_dir, "best_3d_unet.pth"))
                break

end_time = time.time()
writer.close()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training Elapsed time: {minutes} minutes {seconds} seconds")


plt.figure("2D U-Net", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
x2 = [i + 1 for i in range(len(epoch_val_loss_values))]
y1 = epoch_loss_values
y2 = epoch_val_loss_values
y2_interp = np.interp(x, x2, y2)
plt.xlabel("epoch")
plt.plot(x, y1, label='training')
plt.plot(x, y2, label='validation')
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.tight_layout()
plt.show()

plt.figure("2D U-Net1", (12, 6))
plt.subplot(1, 2, 1)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlim(1, len(epoch_loss_values))
ax1.plot(x, y1, color='b', label='Training Loss')
ax2.plot(x2, y2, color='r', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Validation Loss')
plt.title("Epoch Average Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.tight_layout()
plt.show()

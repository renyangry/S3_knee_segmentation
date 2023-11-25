from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    Activations,
)
from monai.networks.nets import UNet, VNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import glob
import time
import random

from torch.utils.tensorboard import SummaryWriter
from ct_all_preprocess.utils import *



data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
data_dir = os.path.join(data_root, which_dataset, 'VNet')
check_path_exist(data_dir)
all_train_images = sorted(glob.glob(os.path.join(data_root, which_dataset, "seperated_leg_img", "*.nii.gz")))
all_train_labels = (sorted(glob.glob(os.path.join(data_root, which_dataset, "seperated_leg_seg", "left_*.nii.gz")))) + (
    sorted(glob.glob(os.path.join(data_root, which_dataset, "seperated_leg_seg", "right_*.nii.gz"))))
# check_num_of_pt_4eval(all_train_labels, all_train_images)

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

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-4], data_dicts[-4:]


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-400,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(64, 64, 64),
            # translate_range=(15, 15, 15),
            rotate_range=(np.pi / 9, np.pi / 9, np.pi / 9),
            scale_range=(0.25, 0.25, 0.25)),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-400,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)

# check_ds = Dataset(data=train_files, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=1)
# check_data = first(check_loader)
# image, label = (check_data["image"][0][0], check_data["label"][0][0])
# # print(f"image shape: {image.shape}, label shape: {label.shape}")
# # plot a random slice
# random_integer = random.randint(0, 48)
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, random_integer], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:, :, random_integer])
# plt.show()

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

device = torch.device("cuda:0")
#version 1 - 3D U-Net
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
loss_function = DiceLoss(sigmoid=True)
# nn.BCELoss()
# DiceLoss(sigmoid=True)
# DiceCELoss(sigmoid=True)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric()

######################################start training############################################
max_epochs = 5000
val_interval = 5
best_metric = -1
best_metric_epoch = -1
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 5
epoch_loss_values = []
epoch_val_loss_values = []
metric_values = []
# post_label = Compose([AsDiscrete()])
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
writer = SummaryWriter(data_dir)

start_time = time.time()


for epoch in range(max_epochs):
    print("-" * 10)
    # print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("epoch_loss", epoch_loss, epoch + 1)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (64, 64, 64)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_loss = loss_function(val_outputs, val_labels)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                epoch_val_loss += val_loss.item()

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            writer.add_scalar("val_mean_dice", metric, epoch + 1)

            epoch_val_loss /= len(val_ds)
            epoch_val_loss_values.append(epoch_val_loss)
            print(f"epoch {epoch + 1} average val loss: {epoch_val_loss:.4f}")
            writer.add_scalar("epoch_val_loss", epoch_val_loss, epoch + 1)

            dice_metric.reset()

            if epoch_val_loss > epoch_loss:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                # torch.save(model.state_dict(), os.path.join(data_dir, "epoch_" + str(epoch + 1) + "_3d_unet.pth"))
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

            if epochs_no_improve == patience:
                print(f"early stopping at current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}")
                torch.save(model.state_dict(), os.path.join(data_dir, "best_3d_unet.pth"))
                break

            # plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
            # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

writer.close()

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training Elapsed time: {minutes} minutes {seconds} seconds")


plt.figure("3D U-Net", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
x2 = [val_interval * (i + 1) for i in range(len(epoch_val_loss_values))]
y1 = epoch_loss_values
y2 = epoch_val_loss_values
plt.xlabel("epoch")
plt.plot(x2, y1[::val_interval], label='training')
plt.plot(x2, y2, label='validation')
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(x2, metric_values)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(data_dir, "3d_net_training.png"))

# x = [i + 1 for i in range(len(epoch_loss_values))]
# x2 = [val_interval * (i + 1) for i in range(len(epoch_val_loss_values))]
# y1 = epoch_loss_values
# y2 = epoch_val_loss_values
# plt.figure("3D U-Net1", (12, 6))
# plt.subplot(1, 2, 1)
# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax1.set_xlim(1, len(epoch_loss_values))
# ax1.plot(x, y1, color='b', label='Training Loss')
# ax2.plot(x2, y2, color='r', label='Validation Loss')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Training Loss')
# ax2.set_ylabel('Validation Loss')
# plt.title("Epoch Average Loss")
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# plt.xlabel("epoch")
# plt.plot(x2, metric_values)
# plt.tight_layout()
# plt.show()


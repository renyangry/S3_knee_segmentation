
import torch
import json

from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from label_set_loss_functions.loss import LeafDiceCE

nnUNet_raw_data_base="/home/rgu/Documents/UK dataset/nnUNet_raw"
nnUNet_preprocessed="/home/rgu/Documents/UK dataset/nnUNet_raw/preprocessed"
nnUNet_results="/home/rgu/Documents/UK dataset/nnUNet_raw/results"


class nnUNetTrainerV2_leafDCE(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        self.num_epochs = 500
        labels_superset_map = {3:[1,2]}
        self.loss = LeafDiceCE(labels_superset_map)
        super(nnUNetTrainerV2_leafDCE, self).__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        


# S3 Knee Segmentation — Supporting Code

This repository contains the code supporting the following publication:

> **[Title available at the journal]**  
> *The Knee*, 2026. DOI: [10.1016/S0968-0160(26)00206-1](https://www.thekneejournal.com/article/S0968-0160(26)00206-1/fulltext)

The code implements a semi-supervised pipeline for automated femur and tibia segmentation from standard full-leg CT images, followed by pre-operative surgical planning.

---

## Pipeline Overview

```
Full-leg CT (DICOM)
       │
       ▼
1. CT Preprocessing
   - DICOM → NIfTI conversion with dcm2niix
   - Split full-leg image into left/right single-leg volumes
       │
       ▼
2. Partially-Supervised Segmentation (nnUNet)
   supervised/
   - Dataset preparation with partial labels (femur=1, tibia=2)
   - nnUNet training and inference
       │
       ▼
3. Surgical Planning
   surgical_planning/
   - Distal femur condyle landmark detection
   - Automatic implant centre localisation
```
---

## Repository Structure

```
S3_knee_segmentation/
│
├── supervised/                 # Partially-supervised nnUNet segmentation
│   ├── nnunet_NMDID_preprocess.py       # Dataset preprocessing (NMDID)
│   ├── nnunet_NMDID_combine2ds.sh       # Combine two datasets
│   ├── nnunet_ps_generate_json.py       # Generate nnUNet dataset.json
│   ├── nnunet_ps_preprocessing_relabel.py  # Relabel for partial supervision
│   ├── nnunet_ps_postprocessing.py      # Post-process nnUNet predictions
│   ├── nnunet_training.sh               # nnUNet training commands
│   └── nnunet_eval4paper.py             # Compute DSC/HD metrics for paper
│
├── surgical_planning/          # Pre-operative implant planning
│   ├── surgical_plan_final.py  # Main planning: condyle landmarks + implant centre
│   ├── surgical_plan_register.py   # Registration-based planning variant
│
├── requirements.txt
└── Dockerfile
```

---

## Installation

```bash
git clone https://github.com/renyangry/S3_knee_segmentation.git
cd S3_knee_segmentation
pip install -r requirements.txt
```

nnUNet must be installed separately following the [nnUNetv2 installation guide](https://github.com/MIC-DKFZ/nnUNet).

---

## Usage

### 1. CT Preprocessing

Convert DICOM to NIfTI and split full-leg images:

```bash
# DICOM → NIfTI
bash ct_dcm2nifti.sh

# Split full-leg CT into left/right single-leg volumes
python image_split_save.py

# Or run the full automated pipeline
bash automatic_seg_pipeline.sh
```

### 2. Supervised Segmentation (nnUNet)

```bash
# Prepare dataset with partial labels
python supervised/nnunet_ps_preprocessing_relabel.py
python supervised/nnunet_ps_generate_json.py

# Train nnUNet
bash supervised/nnunet_training.sh

# Evaluate
python supervised/nnunet_eval4paper.py
```

### 3. Surgical Planning

```bash
python surgical_planning/surgical_plan_final.py
```

---

## Citation

If you use this code, please cite:

```
@article{s3_knee_2026,
  title   = {A Clinical Applicable Study on Lower Limb Segmentation From CT Images for Total Knee Arthroplasty},
  journal = {The Knee},
  year    = {2026},
  doi     = {10.1016/S0968-0160(26)00206-1}
}
```

---

## License

See [LICENSE](LICENSE) for details.

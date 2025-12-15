# Multimodal Surgical Tracking Dataset — Companion Code

This repository provides acquisition and preprocessing scripts for a multimodal dataset recorded for motion-aware surgical tracking and navigation research (multi-view NIR video + ultrasound + CT-derived assets).  
The dataset files are released on Zenodo: **[ZENODO_DOI_OR_URL]**.

---

## What's in this repo

### Python (acquisition + preprocessing)
- `multithreaded_capture.py` — Multi-threaded capture utility (e.g., synchronized camera capture).
- `preprocess_all.py` — Batch preprocessing entry point.
- `mainfunctions.py` — Shared helper functions.
- `read_and_process_ultrasonic.py` — Example pipeline to read and process ultrasound data.

### MATLAB (ultrasound utilities)
- `ULAOP_UnpackProcessor.m` — Unpacking/processing utilities for ULA-OP exports.
- `readBeamformedDataULAOP.m` — Reader for beamformed data.
- `setUpReconstructionStructure.m` — Reconstruction configuration helper.
- `applyScanConversion.m` — Scan conversion helper.
- `scanSequencer.m`, `aperture.m`, `Probe.m` — Probe/sequence/aperture utilities.
- `packDataset.m`, `packBfDataset.m`, `map_multi_acquisition_files.m` — Dataset packaging/mapping helpers.

---

## Requirements

### Python
- Python ≥ 3.9
- Recommended packages: `numpy`, `scipy`, `opencv-python`, `tqdm`, `matplotlib`
- If you capture from Allied Vision / Vimba: install the vendor SDK and Python bindings as shown in the vendor documentation.

Install common dependencies:
pip install numpy scipy opencv-python tqdm matplotlib

### MATLAB
- MATLAB R2020b+ recommended
- If you rely on additional toolboxes (e.g., Signal Processing), ensure they are installed.

---

## Getting the dataset

1. Download the dataset parts from Zenodo: **[ZENODO_DOI_OR_URL]**
2. Reconstruct locally (example if the dataset was released in multiple parts):

cat dataset_part_* > dataset.tar.gz
tar -xzf dataset.tar.gz

3. Confirm you have the expected structure described in the paper (e.g., `sessions/`, `CT_scans/`, `mesh_models/`, `calibrations/`).

---

## Quick start

### 1) Preprocess everything (Python)
Typical usage is to point the script at the dataset root:
python preprocess_all.py --data_root /path/to/HoloWrist_Dataset --out_root /path/to/output

If your scripts use different arguments, run:
python preprocess_all.py --help


### 2) Read/process ultrasound data (Python)
python read_and_process_ultrasonic.py --data_root /path/to/HoloWrist_Dataset --session sessions/in_vivo/session_01


### 3) Ultrasound processing (MATLAB)
From MATLAB, add the repo folder to the path:

addpath(genpath(pwd));



Example workflow (adapt to your dataset paths):
dataRoot = '/path/to/HoloWrist_Dataset';
sessionDir = fullfile(dataRoot, 'sessions', 'in_vivo', 'session_01', 'ultrasound');

% Example: read beamformed data
bf = readBeamformedDataULAOP(sessionDir);

% Example: scan conversion (if needed)
img = applyScanConversion(bf);
imshow(img, []);

---

## Typical pipelines

### NIR video
- Load raw PNG frames from `sessions/.../nir/raw/`
- Optionally run enhancement / normalization (if included in your preprocessing scripts)
- Export enhanced frames to `sessions/.../nir/enhanced/` or your chosen output directory

### Ultrasound
- Read raw/beamformed exports from `sessions/.../ultrasound/raw/`
- Unpack/process to B-mode representations
- Optionally generate `left_images/` and `right_images/` (two arrays)

---

## Outputs and file formats
- **NIR**: 8-bit grayscale PNG images
- **Ultrasound**: raw/beamformed data (often `.mat`) + optional PNG B-mode images
- **CT**: `.nii.gz` volumes + segmentations (bones/skin)
- **Meshes**: `.stl`

---

## Reproducibility notes
Acquisition scripts assume your hardware/software setup (camera SDK, trigger configuration, etc.). These scripts are provided primarily for transparency and reproducibility of the released dataset. If your environment differs, hardware-related sections may require adaptation.

---

## Citation
If you use this dataset or code, please cite:
- **The dataset (Zenodo)**: `[ZENODO_CITATION]`
- **The paper**: `[PAPER_CITATION]`

---

## License
- **Code**: `[CODE_LICENSE]`  
- **Dataset**: see Zenodo record license.

---

## Contact
For questions or issues: **[CONTACT_EMAIL]**  
Or open a GitHub Issue with a minimal reproduction snippet and log output.

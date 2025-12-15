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

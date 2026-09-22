# SAM-FT-HN

**Coarse-to-Fine Adaptation of a Volumetric Segment Anything Model for
Automatic Head-and-Neck Organ-at-Risk Segmentation**

SAM-FT-HN is a 3D medical image segmentation framework for automatic
segmentation of head-and-neck organs at risk (OARs) from CT volumes. The
framework combines a coarse localization network with automatic
anatomical prompt generation, a fine-tuned volumetric SAM backbone,
tiny-organ ROI refinement, and organ-specific threshold calibration.

## Repository Structure

```text
SAM-FT-HN/
├── assets/
│   └── SAM_FT_HN_framework.png
│
├── data/
│   ├── imagesTr/
│   ├── labelsTr/
│   ├── imagesVal/
│   └── labelsVal/
│
├── models/
│   ├── coarse_localization/
│   │   ├── __init__.py
│   │   ├── unet3d.py
│   │   └── coarse_model.py
│   │
│   └── sam_med3d/
│       ├── __init__.py
│       ├── build_sam3D.py
│       └── modeling/
│           ├── image_encoder3D.py
│           ├── prompt_encoder3D.py
│           ├── mask_decoder3D.py
│           ├── sam3D.py
│           └── transformer3D.py
│
├── datasets/
│   ├── data_paths.py
│   ├── dataset.json
│   └── dataloader.py
│
├── training/
│   ├── train_coarse.py
│   ├── train_sam_ft_hn.py
│   ├── losses.py
│   └── trainer.py
│
├── inference/
│   ├── pipeline.py
│   ├── predict_case.py
│   └── batch_inference.py
│
├── utils/
│   ├── io.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── visualization.py
│
├── checkpoints/
│   └── sam_ft_hn/
│
├── scripts/
│   ├── train_coarse.sh
│   ├── train_sam.sh
│   └── inference.sh
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Install Conda

We recommend using Conda to manage the Python environment.

If Conda is not already installed, install Miniconda or Anaconda first.

Confirm that Conda is available:

conda --version
3. Create the SAM-FT-HN Environment

Create a new environment with Python 3.10:

conda create -n sam_ft_hn python=3.10 -y

Activate the environment:

conda activate sam_ft_hn

Verify the Python version:

python --version

The output should indicate Python 3.10.x.

4. Upgrade pip

Before installing the project dependencies, update pip:

python -m pip install --upgrade pip
5. Install PyTorch

SAM-FT-HN is designed to run with GPU acceleration. Install a PyTorch build that is compatible with the CUDA environment on your system.

First, check your NVIDIA GPU and driver:

nvidia-smi

Then install the appropriate PyTorch version for your CUDA configuration.

After installation, verify that PyTorch can detect the GPU:

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

A successful GPU installation should report:

CUDA available: True
GPU: <your NVIDIA GPU>

Note: The exact PyTorch installation command depends on the CUDA/PyTorch versions used in your environment. For reproducibility, use the versions specified in requirements.txt when available.

6. Install SAM-FT-HN Dependencies

From the root directory of the repository, run:

pip install -r requirements.txt

This installs the Python packages required for training, inference, medical-image processing, and evaluation.

7. Install SAM-Med3D Backbone

SAM-FT-HN uses SAM-Med3D as its volumetric Segment Anything backbone.

The required SAM-Med3D components should be available under:

### Model Architecture

```text
models/
└── sam_med3d/
    ├── __init__.py
    ├── build_sam3D.py
    └── modeling/
        ├── image_encoder3D.py
        ├── prompt_encoder3D.py
        ├── mask_decoder3D.py
        ├── sam3D.py
        ├── transformer3D.py
        └── ...
```

If these files are already included in the repository, no additional source-code installation is required.

When SAM-Med3D source code is reused or modified, follow the license and citation requirements of the original SAM-Med3D project.

8. Download the Pretrained SAM-Med3D Checkpoint

Download the pretrained SAM-Med3D checkpoint from the original SAM-Med3D project.

Create the checkpoint directory if necessary:

mkdir -p checkpoints/sam_med3d

Place the downloaded checkpoint in:

```text
SAM-FT-HN/
└── checkpoints/
    └── exp/
        └── sam_model_dice_best.pth
```
The resulting path should therefore be:

checkpoints/exp/sam_model_dice_best.pth

Update the checkpoint path in the corresponding SAM-FT-HN configuration file if your checkpoint is stored elsewhere.

9. Verify the Installation

Check that the main dependencies can be imported:

python -c "import torch; import numpy; import nibabel; print('Basic dependencies loaded successfully.')"

You can also verify CUDA support:

python -c "import torch; print(torch.cuda.is_available())"

For a GPU environment, the expected output is:

True
10. Prepare the Dataset Directories

Create the expected dataset structure:
```text
mkdir -p data/imagesTr
mkdir -p data/labelsTr
mkdir -p data/imagesVal
mkdir -p data/labelsVal
mkdir -p data/splits
```
The directory should look like:
```text
data/
├── imagesTr/
├── labelsTr/
├── imagesVal/
├── labelsVal/
└── splits/
```
Place training CT volumes in imagesTr/ and their corresponding segmentation masks in labelsTr/.

Place validation CT volumes in imagesVal/ and their corresponding masks in labelsVal/.

For example:
```text
data/
├── imagesTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
│
├── labelsTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
│
├── imagesVal/
├── labelsVal/
└── splits/
```
CT and label volumes for the same patient should use consistent filenames.


---

## SAM-Med3D Backbone

SAM-FT-HN builds on the volumetric SAM-Med3D architecture. The SAM
backbone is kept modular so that the image encoder, prompt encoder, and
mask decoder can be loaded from pretrained weights and selectively
fine-tuned.

Update the checkpoint path in the corresponding configuration file.

Original SAM-Med3D project:

https://github.com/uni-medical/SAM-Med3D

Please follow the original repository's license and citation
requirements when redistributing or adapting its code.

---

## Dataset Organization

A recommended dataset layout is:

```text
data/
├── imagesTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
|
├── labelsTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
|
├── imagesVal/
├── labelsVal/
└── splits/
```

Each CT volume should have a corresponding segmentation label volume.
Preprocessing should be consistent across training, validation, and
testing.

Do not include protected patient information in the public repository.

---

## Head-and-Neck OARs

The framework can be configured for the following OARs:

```text
Brain
Brainstem
Esophagus
L_brachialplexus
R_brachialplexus
L_eye
R_eye
L_parotid
R_parotid
L_submandible
R_submandible
Larynx
Lips
Mandible
Oralcavity
Spinalcord
L_lens
R_lens
Chiasm
L_cochlea
R_cochlea
L_opticnerve
R_opticnerve
```

Modify the organ list in the configuration files if a different label
set is used.

---

## Outputs

A typical inference directory may contain:

```text
predictions/
└── case_001/
    ├── Brain.nii.gz
    ├── Brainstem.nii.gz
    ├── L_lens.nii.gz
    ├── R_lens.nii.gz
    ├── Chiasm.nii.gz
    └── ...
```
---

## Reproducibility

For reproducible experiments, record:

- Dataset split
- Random seed
- CT preprocessing parameters
- Input spacing and volume/patch size
- SAM-Med3D checkpoint
- Frozen/trainable network components
- Optimizer and learning rate
- Number of epochs
- Loss functions
- Prompt-generation strategy
- Tiny-organ ROI settings
- Organ-specific thresholds

All parameters used for the final test evaluation should be determined
without access to the held-out test labels.
````

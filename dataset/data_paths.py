import os
from glob import glob
PROJ_DIR = r"H:\Research\SAM-FT-HN"
img_datas = glob(os.path.join(PROJ_DIR, "data_rap", "medical_preprocessed", "*", "*"))

all_classes = [
    "brain",
    "brainstem",
    "esophagus",
    "l_brachialplexus",
    "l_eye",
    "l_parotid",
    "l_submandible",
    "larynx",
    "lips",
    "mandible",
    "oralcavity",
    "r_brachialplexus",
    "r_eye",
    "r_parotid",
    "r_submandible",
    "spinalcord",
    "l_lens",
    "r_lens",
    "chiasm",
    "l_cochlea",
    "r_cochlea",
    "l_opticnerve",
    "r_opticnerve"
]

all_datasets = [
'data_rap',
]
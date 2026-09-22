import os.path as osp
import os
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio


def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None,
                 mode="linear"):

    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    processed_image = resampled_subject.img

   
    if n is not None:
        tensor_data = processed_image.data
        if isinstance(n, int):
            n = [n]
      
        mask = torch.zeros_like(tensor_data, dtype=torch.bool)
        for ni in n:
            mask = mask | (tensor_data == ni)
        tensor_data = torch.where(mask, 1, 0).float()
        processed_image = tio.ScalarImage(tensor=tensor_data, affine=processed_image.affine)

      
    if reference_image is not None:
        reference_size = reference_image.shape[1:]  
        cropper_or_padder = tio.CropOrPad(reference_size)
        processed_image = cropper_or_padder(processed_image)

    processed_image.save(output_path)

dataset_root = r"H:\Research\SAM-FT-HN"
datasets = ["data_rap"]  
target_dir = osp.join(dataset_root, "medical_preprocessed40")

for dataset in datasets:
    dataset_dir = osp.join(dataset_root, dataset)
    img_dir = osp.join(dataset_dir, "imagesTr")
    label_dir = osp.join(dataset_dir, "labelsTr")
    resample_cache = osp.join(dataset_dir, "imagesTr_1.5")
    os.makedirs(resample_cache, exist_ok=True)

    print(f"process: {dataset}")

    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]

    for img_file in tqdm(img_files, desc="image"):
 
        case_id = img_file.split('_0000.')[0]
        case_label_dir = osp.join(label_dir, case_id)

        if not osp.exists(case_label_dir):
            continue

            
        img_path = osp.join(img_dir, img_file)
        resampled_img_path = osp.join(resample_cache, img_file)

        if not osp.exists(resampled_img_path):
            resample_nii(img_path, resampled_img_path)

        reference_img = tio.ScalarImage(resampled_img_path)

        
        organ_files = [f for f in os.listdir(case_label_dir)
                       if f.endswith('.nii.gz') and not f.startswith('.')]

        for organ_file in organ_files:
            
            organ_name = osp.splitext(osp.splitext(organ_file)[0])[0]

           
            target_organ_dir = osp.join(target_dir, organ_name, 'rap')
            target_img_dir = osp.join(target_organ_dir, "imagesTr")
            target_label_dir = osp.join(target_organ_dir, "labelsTr")
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_label_dir, exist_ok=True)

           
            target_img_path = osp.join(target_img_dir, f"{case_id}.nii.gz")
            target_label_path = osp.join(target_label_dir, f"{case_id}.nii.gz")

           
            if not osp.exists(target_img_path):
                shutil.copy(resampled_img_path, target_img_path)

           
            label_path = osp.join(case_label_dir, organ_file)

            
            label_img = nib.load(label_path)
            spacing = label_img.header['pixdim'][1:4]
            voxel_vol = spacing[0] * spacing[1] * spacing[2]
            label_data = label_img.get_fdata()
            organ_vol = label_data.sum() * voxel_vol

            
            if organ_vol < 10:
                print(f"process: {organ_name} ({organ_vol:.2f}mm³)")
                continue

                
            resample_nii(
                label_path,
                target_label_path,
                reference_image=reference_img,
                mode="nearest"
            )

print("done")

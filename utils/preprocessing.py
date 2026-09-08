# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet_v2.py
@Time    :   2025/06/12 01:07:39
@Author  :   Assistant
@Contact :   user@example.com
@Brief   :   处理多标签分离结构的nnUNet数据集，适配SAM-Med3D格式
'''

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
    """
    重采样nii.gz 文件并可选进行二值化处理

    参数：
    - input_path: 输入文件路径
    - output_path: 输出文件路径
    - target_spacing: 目标分辨率(默认1.5mm各向同性)
    - n: 需要提取的标签索引(单个或列表)
    - reference_image: 空间参考图像(torchio对象)
    - mode: 插值方式(linear/nearest)
    """
    # 加载并重采样图像
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    processed_image = resampled_subject.img

    # 标签二值化处理
    if n is not None:
        tensor_data = processed_image.data
        if isinstance(n, int):
            n = [n]
        # 创建二值掩码
        mask = torch.zeros_like(tensor_data, dtype=torch.bool)
        for ni in n:
            mask = mask | (tensor_data == ni)
        tensor_data = torch.where(mask, 1, 0).float()
        processed_image = tio.ScalarImage(tensor=tensor_data, affine=processed_image.affine)

        # 空间对齐
    if reference_image is not None:
        reference_size = reference_image.shape[1:]  # 去除通道维度
        cropper_or_padder = tio.CropOrPad(reference_size)
        processed_image = cropper_or_padder(processed_image)

    processed_image.save(output_path)

"""
修改路径即可
"""
dataset_root = r"H:\Research\SAM-FT-HN"
datasets = ["data_rap"]  # 可扩展多个数据集
target_dir = osp.join(dataset_root, "medical_preprocessed40")

for dataset in datasets:
    dataset_dir = osp.join(dataset_root, dataset)
    img_dir = osp.join(dataset_dir, "imagesTr")
    label_dir = osp.join(dataset_dir, "labelsTr")
    resample_cache = osp.join(dataset_dir, "imagesTr_1.5")
    os.makedirs(resample_cache, exist_ok=True)

    print(f"处理数据集: {dataset}")

    # 获取所有图像文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]

    for img_file in tqdm(img_files, desc="处理图像"):
        # 解析案例ID (e.g., word_0001)
        case_id = img_file.split('_0000.')[0]
        case_label_dir = osp.join(label_dir, case_id)

        if not osp.exists(case_label_dir):
            continue

            # 处理图像重采样
        img_path = osp.join(img_dir, img_file)
        resampled_img_path = osp.join(resample_cache, img_file)

        if not osp.exists(resampled_img_path):
            resample_nii(img_path, resampled_img_path)

        reference_img = tio.ScalarImage(resampled_img_path)

        # 获取器官标签文件
        organ_files = [f for f in os.listdir(case_label_dir)
                       if f.endswith('.nii.gz') and not f.startswith('.')]

        for organ_file in organ_files:
            # 提取器官名称 (e.g., brain)
            organ_name = osp.splitext(osp.splitext(organ_file)[0])[0]

            # 创建目标目录
            target_organ_dir = osp.join(target_dir, organ_name, 'rap')
            target_img_dir = osp.join(target_organ_dir, "imagesTr")
            target_label_dir = osp.join(target_organ_dir, "labelsTr")
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_label_dir, exist_ok=True)

            # 设置目标路径
            target_img_path = osp.join(target_img_dir, f"{case_id}.nii.gz")
            target_label_path = osp.join(target_label_dir, f"{case_id}.nii.gz")

            # 复制重采样后的图像
            if not osp.exists(target_img_path):
                shutil.copy(resampled_img_path, target_img_path)

            # 处理器官标签
            label_path = osp.join(case_label_dir, organ_file)

            # 计算原始标签体积
            label_img = nib.load(label_path)
            spacing = label_img.header['pixdim'][1:4]
            voxel_vol = spacing[0] * spacing[1] * spacing[2]
            label_data = label_img.get_fdata()
            organ_vol = label_data.sum() * voxel_vol

            # 体积过滤 (>10mm³)
            if organ_vol < 10:
                print(f"跳过小体积器官: {organ_name} ({organ_vol:.2f}mm³)")
                continue

                # 重采样对齐标签
            resample_nii(
                label_path,
                target_label_path,
                reference_image=reference_img,
                mode="nearest"
            )

print("数据处理完成！")
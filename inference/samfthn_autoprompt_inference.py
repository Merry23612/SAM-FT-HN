import argparse
import json
import os
import os.path as osp
from collections import OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D
from utils.data_loader import Dataset_Union_ALL_Val

def normalize_organ_name(name):
    if name is None:
        return "unknown"
    s = str(name).strip().lower().replace("\\", "/").split("/")[-1]
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    aliases = {
        "left_lens": "l_lens", "right_lens": "r_lens",
        "left_eye": "l_eye", "right_eye": "r_eye",
        "left_parotid": "l_parotid", "right_parotid": "r_parotid",
        "left_submandible": "l_submandible", "right_submandible": "r_submandible",
        "left_submandibular": "l_submandible", "right_submandibular": "r_submandible",
        "left_cochlea": "l_cochlea", "right_cochlea": "r_cochlea",
        "left_opticnerve": "l_opticnerve", "right_opticnerve": "r_opticnerve",
        "left_optic_nerve": "l_opticnerve", "right_optic_nerve": "r_opticnerve",
        "l_optic_nerve": "l_opticnerve", "r_optic_nerve": "r_opticnerve",
        "oral_cavity": "oralcavity", "spinal_cord": "spinalcord",
        "l_brachial_plexus": "l_brachialplexus", "r_brachial_plexus": "r_brachialplexus",
    }
    return aliases.get(s, s)
DEFAULT_ORGAN_THRESHOLDS = {
    "brain": 0.30,
    "brainstem": 0.30,
    "esophagus": 0.45,
    "l_brachialplexus": 0.40,
    "l_eye": 0.50,
    "l_parotid": 0.25,
    "l_submandible": 0.50,
    "larynx": 0.45,
    "lips": 0.40,
    "mandible": 0.20,
    "oralcavity": 0.45,
    "r_brachialplexus": 0.40,
    "r_eye": 0.50,
    "r_parotid": 0.25,
    "r_submandible": 0.50,
    "spinalcord": 0.50,
    "l_lens": 0.35,
    "r_lens": 0.35,
    "chiasm": 0.40,
    "l_cochlea": 0.45,
    "r_cochlea": 0.45,
    "l_opticnerve": 0.40,
    "r_opticnerve": 0.40,
}

SMALL_ORGANS = {
    "l_lens", "r_lens", "chiasm",
    "l_cochlea", "r_cochlea",
    "l_opticnerve", "r_opticnerve",
}
DEFAULT_TINY_ROI_RADIUS_MM = {
    "l_lens": 15.0,
    "r_lens": 15.0,
    "chiasm": 18.0,
    "l_cochlea": 20.0,
    "r_cochlea": 20.0,
    "l_opticnerve": 22.0,
    "r_opticnerve": 22.0,
}
DEFAULT_SMALL_VOLUME_RANGES_CM3 = {
    "l_lens": [0.2, 0.4],
    "r_lens": [0.2, 0.4],
    "chiasm": [0.3, 0.6],
    "l_cochlea": [0.3, 0.5],
    "r_cochlea": [0.3, 0.5],
    "l_opticnerve": [0.4, 0.7],
    "r_opticnerve": [0.4, 0.7],
}


def load_float_map(path, defaults):
    cfg = dict(defaults)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            cfg[normalize_organ_name(k)] = float(v)
    return cfg


def load_volume_ranges(path):
    cfg = {k: [float(v[0]), float(v[1])] for k, v in DEFAULT_SMALL_VOLUME_RANGES_CM3.items()}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if len(v) != 2:
                raise ValueError(f"Volume range for {k} must contain [Vmin, Vmax].")
            cfg[normalize_organ_name(k)] = [float(v[0]), float(v[1])]
    return cfg

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoarseUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=23, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv3D(in_channels, b); self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(b, b*2); self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(b*2, b*4); self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(b*4, b*8)
        self.up3 = nn.ConvTranspose3d(b*8, b*4, 2, stride=2); self.dec3 = DoubleConv3D(b*8, b*4)
        self.up2 = nn.ConvTranspose3d(b*4, b*2, 2, stride=2); self.dec2 = DoubleConv3D(b*4, b*2)
        self.up1 = nn.ConvTranspose3d(b*2, b, 2, stride=2); self.dec1 = DoubleConv3D(b*2, b)
        self.out_conv = nn.Conv3d(b, out_channels, 1)

    @staticmethod
    def _match(x, ref):
        return F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=False) if x.shape[-3:] != ref.shape[-3:] else x

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], 1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], 1))
        return self.out_conv(d1)


def load_coarse_model(path, device):
    ckpt = torch.load(path, map_location=device)
    organs = ckpt.get("organs")
    if organs is None:
        raise RuntimeError("Coarse checkpoint must contain the 'organs' list.")
    state = ckpt.get("model_state_dict", ckpt)
    first_key = next((k for k in state if k.endswith("enc1.block.0.weight")), None)
    base = int(state[first_key].shape[0]) if first_key else 16
    model = CoarseUNet3D(1, len(organs), base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    organ_to_idx = {normalize_organ_name(o): i for i, o in enumerate(organs)}
    patch_size = int(ckpt.get("patch_size", 128))
    stride = int(ckpt.get("stride", 64))
    return model, organ_to_idx, patch_size, stride

def unwrap_meta_value(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return unwrap_meta_value(value[0])
        return [unwrap_meta_value(v) for v in value]
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr.tolist()
    return value


def get_image_path(meta):
    return str(unwrap_meta_value(meta.get("image_path", "unknown")))


def get_spacing(meta):
    sp = unwrap_meta_value(meta.get("spacing", None))
    if sp is None:
        return (1.5, 1.5, 1.5)
    a = np.asarray(sp, dtype=float).reshape(-1)
    return tuple(float(x) for x in a[:3]) if len(a) >= 3 else (1.5, 1.5, 1.5)


def ct_z_normalize_torch(x, hu_min=-1000.0, hu_max=1000.0, body_hu=-900.0):
    """CT-only z-score; no label-derived foreground mask."""
    x = torch.clamp(x.float(), hu_min, hu_max)
    out = torch.empty_like(x)
    for b in range(x.shape[0]):
        body = x[b] > body_hu
        vals = x[b][body] if body.any() else x[b].reshape(-1)
        out[b] = (x[b] - vals.mean()) / vals.std().clamp_min(1e-6)
    return out


def sliding_starts(length, patch, stride):
    if length <= patch:
        return [0]
    starts = list(range(0, length-patch+1, stride))
    if starts[-1] != length-patch:
        starts.append(length-patch)
    return starts


def pad_to_patch(x, patch):
    D,H,W = x.shape[-3:]
    pd,ph,pw = max(0,patch-D),max(0,patch-H),max(0,patch-W)
    return F.pad(x,(0,pw,0,ph,0,pd),value=0),(D,H,W)


@torch.no_grad()
def predict_coarse_probability(image_norm, model, organ_idx, patch_size, stride, device):

    x, original = pad_to_patch(image_norm, patch_size)
    D,H,W = x.shape[-3:]
    prob_sum = torch.zeros((D,H,W), dtype=torch.float32)
    count = torch.zeros((D,H,W), dtype=torch.float32)

    for z in sliding_starts(D,patch_size,stride):
        for y in sliding_starts(H,patch_size,stride):
            for xx in sliding_starts(W,patch_size,stride):
                patch = x[...,z:z+patch_size,y:y+patch_size,xx:xx+patch_size].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    logits = model(patch)
                    prob = torch.sigmoid(logits[:,organ_idx:organ_idx+1])[0,0].float().cpu()
                prob_sum[z:z+patch_size,y:y+patch_size,xx:xx+patch_size] += prob
                count[z:z+patch_size,y:y+patch_size,xx:xx+patch_size] += 1

    prob_sum /= count.clamp_min(1)
    d,h,w = original
    return prob_sum[:d,:h,:w].numpy().astype(np.float32)


def highest_confidence_component(prob, threshold):
    mask = prob >= float(threshold)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    labels, n = ndi.label(mask)
    if n == 1:
        return mask
    best_label, best_score = None, -np.inf
    for lab in range(1, n+1):
        vox = labels == lab
       
        score = float(prob[vox].sum())
        if score > best_score:
            best_score, best_label = score, lab
    return labels == best_label


def probability_centroid(prob):
    p = np.asarray(prob, dtype=np.float64)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return tuple(int(v) for v in np.unravel_index(np.argmax(p),p.shape))
    coords = np.indices(p.shape)
    return tuple(int(np.clip(round((coords[i]*p).sum()/total),0,p.shape[i]-1)) for i in range(3))


def mask_centroid(mask):
    pts = np.argwhere(mask)
    return None if pts.size==0 else tuple(np.round(pts.mean(axis=0)).astype(int))


def mm_to_vox(mm, spacing):
    return tuple(max(1,int(round(float(mm)/max(float(s),1e-6)))) for s in spacing)


def coarse_map_to_auto_prompt(prob, spacing, threshold=0.5, bbox_margin_mm=12.0, fallback_half_size_mm=30.0):
    comp = highest_confidence_component(prob, threshold)
    center = mask_centroid(comp)
    if center is None:
        center = probability_centroid(prob)

    pts = np.argwhere(comp)
    if pts.size:
        margin = np.asarray(mm_to_vox(bbox_margin_mm,spacing))
        lo = np.maximum(pts.min(axis=0)-margin,0)
        hi = np.minimum(pts.max(axis=0)+margin+1,np.asarray(prob.shape))
    else:
        half = np.asarray(mm_to_vox(fallback_half_size_mm,spacing))
        c = np.asarray(center)
        lo = np.maximum(c-half,0)
        hi = np.minimum(c+half+1,np.asarray(prob.shape))
    box = tuple(int(v) for v in (*lo,*hi))
    return center, box, comp.astype(np.uint8)

def extract_fixed_crop(volume, center, crop_size):
    assert volume.ndim == 5
    D,H,W = volume.shape[-3:]
    size = np.asarray([crop_size]*3)
    center = np.asarray(center,dtype=int)
    start = center-size//2
    end = start+size
    src0 = np.maximum(start,0)
    src1 = np.minimum(end,np.asarray([D,H,W]))
    pad0 = np.maximum(-start,0)
    pad1 = np.maximum(end-np.asarray([D,H,W]),0)
    crop = volume[...,src0[0]:src1[0],src0[1]:src1[1],src0[2]:src1[2]]
    crop = F.pad(crop,(int(pad0[2]),int(pad1[2]),int(pad0[1]),int(pad1[1]),int(pad0[0]),int(pad1[0])),value=0)
    return crop, {
        "src_start": tuple(int(v) for v in src0),
        "src_end": tuple(int(v) for v in src1),
        "pad_before": tuple(int(v) for v in pad0),
        "requested_start": tuple(int(v) for v in start),
        "crop_size": int(crop_size),
    }


def global_to_local(point, info):
    q = np.asarray(point,dtype=int)-np.asarray(info["requested_start"],dtype=int)
    q = np.clip(q,0,info["crop_size"]-1)
    return tuple(int(v) for v in q)


def global_box_to_local(box, info):
    z0,y0,x0,z1,y1,x1 = box
    p0 = global_to_local((z0,y0,x0),info)
    p1 = global_to_local((z1-1,y1-1,x1-1),info)
    return tuple(int(v) for v in (*p0,*p1))


def paste_crop_to_full(crop_prob, info, full_shape):
    out = np.zeros(full_shape,dtype=np.float32)
    ss,se,pb = info["src_start"],info["src_end"],info["pad_before"]
    dz,dy,dx = se[0]-ss[0],se[1]-ss[1],se[2]-ss[2]
    valid = crop_prob[pb[0]:pb[0]+dz,pb[1]:pb[1]+dy,pb[2]:pb[2]+dx]
    out[ss[0]:se[0],ss[1]:se[1],ss[2]:se[2]] = valid
    return out

def encode_auto_prompt(prompt_encoder, points, labels, boxes, masks):
    candidates = [boxes, boxes.view(boxes.shape[0],2,3)]
    for b in candidates:
        try:
            return prompt_encoder(points=[points,labels],boxes=b,masks=masks)
        except (TypeError,RuntimeError,ValueError,AssertionError):
            pass
    return prompt_encoder(points=[points,labels],boxes=None,masks=masks)


@torch.no_grad()
def sam_auto_predict_crop(image_crop_norm, sam_model, point_zyx, box_zyxzyx, prev_prob, device):
    image_crop_norm = image_crop_norm.to(device)
    B = image_crop_norm.shape[0]
    points = torch.tensor([[point_zyx]],dtype=torch.float32,device=device)
    labels = torch.ones((B,1),dtype=torch.float32,device=device)
    boxes = torch.tensor([box_zyxzyx],dtype=torch.float32,device=device)

    low_shape = (image_crop_norm.shape[-1]//4,)*3
    if prev_prob is None:
        low_prompt = torch.zeros((B,1,*low_shape),dtype=torch.float32,device=device)
    else:
        prev = torch.from_numpy(prev_prob)[None,None].float().to(device)
        low_prompt = F.interpolate(prev,size=low_shape,mode="trilinear",align_corners=False)

    emb = sam_model.image_encoder(image_crop_norm)
    sparse,dense = encode_auto_prompt(sam_model.prompt_encoder,points,labels,boxes,low_prompt)
    low_res,_ = sam_model.mask_decoder(
        image_embeddings=emb,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    logits = F.interpolate(low_res,size=image_crop_norm.shape[-3:],mode="trilinear",align_corners=False)
    return torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)
def create_spherical_roi(shape,center,radius_mm,spacing):
    radii = [max(1.0,float(radius_mm)/max(float(s),1e-6)) for s in spacing]
    z,y,x = np.ogrid[:shape[0],:shape[1],:shape[2]]
    cz,cy,cx = center
    d = ((z-cz)/radii[0])**2 + ((y-cy)/radii[1])**2 + ((x-cx)/radii[2])**2
    return d <= 1.0


def centered_box(center,shape,half_vox):
    c,half,shp = np.asarray(center),np.asarray(half_vox),np.asarray(shape)
    lo = np.maximum(c-half,0)
    hi = np.minimum(c+half+1,shp)
    return tuple(int(v) for v in (*lo,*hi))


def robust_prediction_centroid(prob,threshold):
    comp = highest_confidence_component(prob,threshold)
    c = mask_centroid(comp)
    return c if c is not None else probability_centroid(prob)


def tiny_organ_refinement(full_image_norm,first_prob_full,organ,sam_model,spacing,crop_size,base_threshold,radius_cfg,device):
    radius_mm = float(radius_cfg[organ])
    center = robust_prediction_centroid(first_prob_full,base_threshold)
    tight_box = centered_box(center,first_prob_full.shape,mm_to_vox(radius_mm,spacing))

    crop,info = extract_fixed_crop(full_image_norm,center,crop_size)
    point_local = global_to_local(center,info)
    box_local = global_box_to_local(tight_box,info)

    prev_t = torch.from_numpy(first_prob_full)[None,None].float()
    prev_crop,_ = extract_fixed_crop(prev_t,center,crop_size)
    refined_crop = sam_auto_predict_crop(crop,sam_model,point_local,box_local,prev_crop[0,0].numpy(),device)
    refined_full = paste_crop_to_full(refined_crop,info,first_prob_full.shape)

    roi = create_spherical_roi(first_prob_full.shape,center,radius_mm,spacing)
    refined_full *= roi.astype(np.float32)
    return refined_full,center,tight_box

def largest_component(mask):
    mask = np.asarray(mask,dtype=bool)
    if not mask.any():
        return mask
    lab,n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == counts.argmax()


def volume_cm3(mask,spacing):
    return float(mask.sum()) * float(np.prod(spacing)) / 1000.0


def adaptive_threshold_with_training_volume(prob,spacing,base_threshold,vmin_cm3,vmax_cm3,
                                            threshold_min=0.10,threshold_max=0.90,iterations=12):
    base = float(np.clip(base_threshold,threshold_min,threshold_max))

    def eval_thr(t):
        m = largest_component(prob >= t)
        return m, volume_cm3(m,spacing)

    base_mask,base_vol = eval_thr(base)
    if vmin_cm3 <= base_vol <= vmax_cm3:
        return base,base_mask,base_vol

    lo,hi = float(threshold_min),float(threshold_max)
    best_t,best_m,best_v = base,base_mask,base_vol
   
    def interval_error(v):
        if v < vmin_cm3:
            return vmin_cm3-v
        if v > vmax_cm3:
            return v-vmax_cm3
        return 0.0
    best_err = interval_error(base_vol)
    t = base

    for _ in range(iterations):
        m,v = eval_thr(t)
        err = interval_error(v)
        if err < best_err:
            best_t,best_m,best_v,best_err = t,m,v,err
        if err == 0:
            return float(t),m,float(v)
        if v > vmax_cm3:
            lo = max(lo,t)  
            t = (t+hi)/2.0
        else:
            hi = min(hi,t)       
            t = (lo+t)/2.0

    return float(best_t),best_m,float(best_v)

def save_numpy_to_nifti(arr,out_path,meta):
    arr = np.asarray(arr).squeeze()
    ori_arr = np.transpose(arr,(2,1,0))
    out = sitk.GetImageFromArray(ori_arr)

    def vals(key):
        v = unwrap_meta_value(meta.get(key,None))
        return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]

    origin,direction,spacing = vals("origin"),vals("direction"),vals("spacing")
    if origin is not None and len(origin)>=3: out.SetOrigin(origin[:3])
    if direction is not None and len(direction)>=9: out.SetDirection(direction[:9])
    if spacing is not None and len(spacing)>=3: out.SetSpacing(spacing[:3])
    sitk.WriteImage(out,str(out_path))


def compute_dice(gt,pred):
    den = gt.sum()+pred.sum()
    return float("nan") if den==0 else float(2*np.logical_and(gt,pred).sum()/den)


def compute_iou(pred,gt):
    den = np.logical_or(pred,gt).sum()
    return float("nan") if den==0 else float(np.logical_and(pred,gt).sum()/den)


def surface_voxels(mask):
    if not mask.any(): return np.zeros_like(mask,dtype=bool)
    er = binary_erosion(mask,structure=np.ones((3,3,3),dtype=bool),border_value=0)
    return mask & (~er)


def compute_hd95_mm(gt,pred,spacing):
    if not gt.any() or not pred.any(): return float("nan")
    sg,sp = surface_voxels(gt),surface_voxels(pred)
    if not sg.any() or not sp.any(): return float("nan")
    d1 = distance_transform_edt(~sp,sampling=spacing)[sg]
    d2 = distance_transform_edt(~sg,sampling=spacing)[sp]
    return float(np.percentile(np.concatenate([d1,d2]),95))


def infer_organ_from_path(path):
    parts = str(path).replace("\\","/").split("/")
    for token in ("imagesTs","imagesTr","imagesVal"):
        if token in parts:
            i = parts.index(token)
            if i >= 2: return normalize_organ_name(parts[i-2])
    return normalize_organ_name(parts[-4]) if len(parts)>=4 else "unknown"


def make_output_path(root,organ,img_path):
    p = Path(str(img_path).replace("\\","/"))
    dataset = p.parent.parent.name
    out = Path(root)/organ/dataset
    out.mkdir(parents=True,exist_ok=True)
    name = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
    return out,name


def get_args():
    p = argparse.ArgumentParser(description="Fully automatic SAM-FT-HN inference")
    p.add_argument("--test_data_path","-tdp",default=r"H:\Research\SAM-FT-HN\medical_preprocessed")
    p.add_argument("--data_type","-dt",default="Ts")
    p.add_argument("--checkpoint_path","-cp",required=True)
    p.add_argument("--coarse_checkpoint",default=r"H:\Research\SAM-FT-HN\coarse_checkpoints\coarse_best.pth")
    p.add_argument("--output_dir",default=r"H:\Research\SAM-FT-HN\automatic_inference")
    p.add_argument("--model_type","-mt",default="vit_b_ori")
    p.add_argument("--device",default="cuda")
    p.add_argument("--crop_size",type=int,default=128)
    p.add_argument("--coarse_patch_size",type=int,default=128)
    p.add_argument("--coarse_stride",type=int,default=64)
    p.add_argument("--coarse_threshold",type=float,default=0.50)
    p.add_argument("--bbox_margin_mm",type=float,default=12.0)
    p.add_argument("--fallback_half_size_mm",type=float,default=30.0)
    p.add_argument("--hu_min",type=float,default=-1000.0)
    p.add_argument("--hu_max",type=float,default=1000.0)
    p.add_argument("--body_hu",type=float,default=-900.0)
    p.add_argument("--threshold_json",default="",help="validation-selected SAM threshold for each organ")
    p.add_argument("--tiny_roi_radius_json",default="",help="validation-selected tiny-organ radius in mm")
    p.add_argument("--volume_range_json",default="",help="training-derived [Vmin,Vmax] cm3 for small organs")
    p.add_argument("--adaptive_thr_min",type=float,default=0.10)
    p.add_argument("--adaptive_thr_max",type=float,default=0.90)
    p.add_argument("--evaluate_gt",action="store_true")
    p.add_argument("--save_coarse_map",action="store_true")
    p.add_argument("--skip_existing",action="store_true")
    p.add_argument("--split_idx",type=int,default=0)
    p.add_argument("--split_num",type=int,default=1)
    p.add_argument("--seed",type=int,default=2023)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    coarse_model,organ_to_idx,ckpt_patch,ckpt_stride = load_coarse_model(args.coarse_checkpoint,device)
    
    coarse_patch = ckpt_patch or args.coarse_patch_size
    coarse_stride = ckpt_stride or args.coarse_stride
    print(f"Coarse sliding window: {coarse_patch}^3, stride {coarse_stride}")

    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    sam_ckpt = torch.load(args.checkpoint_path,map_location=device)
    sam_state = sam_ckpt.get("model_state_dict",sam_ckpt) if isinstance(sam_ckpt,dict) else sam_ckpt
    sam_model.load_state_dict(sam_state,strict=False)
    sam_model.eval()

    thresholds = load_float_map(args.threshold_json,DEFAULT_ORGAN_THRESHOLDS)
    radii = load_float_map(args.tiny_roi_radius_json,DEFAULT_TINY_ROI_RADIUS_MM)
    volume_ranges = load_volume_ranges(args.volume_range_json)

    os.makedirs(args.output_dir,exist_ok=True)
    with open(osp.join(args.output_dir,"thresholds_used.json"),"w") as f: json.dump(thresholds,f,indent=2)
    with open(osp.join(args.output_dir,"tiny_roi_radius_used.json"),"w") as f: json.dump(radii,f,indent=2)
    with open(osp.join(args.output_dir,"training_volume_ranges_used.json"),"w") as f: json.dump(volume_ranges,f,indent=2)

    all_paths = [p for p in glob(osp.join(args.test_data_path,"*","*")) if osp.isdir(p)]
    dataset = Dataset_Union_ALL_Val(
        paths=all_paths, mode="Val", data_type=args.data_type,
        transform=tio.Compose([tio.ToCanonical()]), threshold=0,
        split_num=args.split_num, split_idx=args.split_idx,
        pcc=False, get_all_meta_info=True,
    )
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    results = OrderedDict()

    for image3D,gt3D_eval,meta in tqdm(loader,desc="Automatic SAM-FT-HN"):
        img_path = get_image_path(meta)
        organ = infer_organ_from_path(img_path)
        if organ not in organ_to_idx:
            print(f"[SKIP] {organ}: not present in coarse checkpoint")
            continue

        out_dir,case_name = make_output_path(args.output_dir,organ,img_path)
        pred_path = out_dir/f"{case_name}_pred_auto.nii.gz"
        if args.skip_existing and pred_path.exists():
            continue

        spacing = get_spacing(meta)
        image_norm = ct_z_normalize_torch(image3D.float(),args.hu_min,args.hu_max,args.body_hu)

        coarse_prob = predict_coarse_probability(
            image_norm,coarse_model,organ_to_idx[organ],coarse_patch,coarse_stride,device
        )
        auto_center,auto_bbox,coarse_component = coarse_map_to_auto_prompt(
            coarse_prob,spacing,args.coarse_threshold,args.bbox_margin_mm,args.fallback_half_size_mm
        )

        if args.save_coarse_map:
            save_numpy_to_nifti(coarse_prob,out_dir/f"{case_name}_coarse_prob.nii.gz",meta)
            save_numpy_to_nifti(coarse_component,out_dir/f"{case_name}_coarse_component.nii.gz",meta)

       
        crop,info = extract_fixed_crop(image_norm,auto_center,args.crop_size)
        point_local = global_to_local(auto_center,info)
        box_local = global_box_to_local(auto_bbox,info)
        first_crop_prob = sam_auto_predict_crop(crop,sam_model,point_local,box_local,None,device)
        full_shape = tuple(int(v) for v in image_norm.shape[-3:])
        first_prob_full = paste_crop_to_full(first_crop_prob,info,full_shape)

        base_thr = float(thresholds.get(organ,0.50))
        refined_center,refined_bbox = None,None
        final_prob = first_prob_full

        if organ in SMALL_ORGANS:
            final_prob,refined_center,refined_bbox = tiny_organ_refinement(
                image_norm,first_prob_full,organ,sam_model,spacing,args.crop_size,
                base_thr,radii,device
            )

        
        final_thr = base_thr
        final_mask = largest_component(final_prob >= final_thr)
        final_volume_cm3 = volume_cm3(final_mask,spacing)

      
        if organ in SMALL_ORGANS:
            if organ not in volume_ranges:
                raise RuntimeError(
                    f"No training-derived volume range configured for small organ {organ}. "
                    "Provide --volume_range_json."
                )
            vmin,vmax = volume_ranges[organ]
            final_thr,final_mask,final_volume_cm3 = adaptive_threshold_with_training_volume(
                final_prob,spacing,base_thr,vmin,vmax,
                args.adaptive_thr_min,args.adaptive_thr_max,
            )

        pred_bin = final_mask.astype(np.uint8)
        save_numpy_to_nifti(pred_bin,pred_path,meta)
        save_numpy_to_nifti(final_prob,out_dir/f"{case_name}_prob_auto.nii.gz",meta)

        info_json = {
            "organ": organ,
            "image_path": img_path,
            "coarse_positive_point_zyx": [int(v) for v in auto_center],
            "coarse_bbox_zyxzyx": [int(v) for v in auto_bbox],
            "coarse_threshold": float(args.coarse_threshold),
            "small_organ": organ in SMALL_ORGANS,
            "tiny_refined_center_zyx": [int(v) for v in refined_center] if refined_center is not None else None,
            "tiny_refined_bbox_zyxzyx": [int(v) for v in refined_bbox] if refined_bbox is not None else None,
            "base_validation_threshold": float(base_thr),
            "final_adaptive_threshold": float(final_thr),
            "final_volume_cm3": float(final_volume_cm3),
            "training_volume_range_cm3": volume_ranges.get(organ),
            "prediction_path": str(pred_path),
        }

    
        if args.evaluate_gt:
            gt = gt3D_eval.detach().cpu().numpy().squeeze().astype(bool)
            if gt.shape != pred_bin.shape:
                if gt.shape == pred_bin.shape[::-1]:
                    gt = np.transpose(gt,(2,1,0))
                else:
                    raise ValueError(f"GT/pred shape mismatch: {gt.shape} vs {pred_bin.shape}")
            pb = pred_bin.astype(bool)
            info_json["dice"] = compute_dice(gt,pb)
            info_json["iou"] = compute_iou(pb,gt)
            info_json["hd95_mm"] = compute_hd95_mm(gt,pb,spacing)

        with open(out_dir/f"{case_name}_autoprompt.json","w",encoding="utf-8") as f:
            json.dump(info_json,f,indent=2)
        results[img_path] = info_json

        print(
            f"\n{organ} | {case_name}\n"
            f"  coarse point: {auto_center}\n"
            f"  coarse box:   {auto_bbox}\n"
            f"  SAM threshold: {base_thr:.3f} -> {final_thr:.3f}\n"
            f"  final volume: {final_volume_cm3:.3f} cm3"
        )

    summary = osp.join(args.output_dir,"automatic_inference_summary.json")
    with open(summary,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2)

    if args.evaluate_gt and results:
        dices = [v["dice"] for v in results.values() if "dice" in v and np.isfinite(v["dice"])]
        hd = [v["hd95_mm"] for v in results.values() if "hd95_mm" in v and np.isfinite(v["hd95_mm"])]
        if dices: print("Mean Dice:",float(np.mean(dices)))
        if hd: print("Mean HD95:",float(np.mean(hd)))
    print("Done. Summary:",summary)


if __name__ == "__main__":
    main()

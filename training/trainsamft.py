import argparse
import datetime
import json
import logging
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from monai.losses import DiceCELoss
from scipy import ndimage as ndi
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--task_name", default="samfthn_auto")
    p.add_argument("--data_root", default=r"H:\Research\SAM-FT-HN\medical_preprocessed")
    p.add_argument("--checkpoint", default="ckpt/sam_med3d.pth", help="SAM-Med3D initialization checkpoint")
    p.add_argument("--coarse_checkpoint", default=r"H:\Research\SAM-FT-HN\coarse_checkpoints\coarse_best.pth")
    p.add_argument("--work_dir", default="work_dir")
    p.add_argument("--model_type", default="vit_b_ori")

    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--step_size", type=int, nargs="+", default=[120, 180])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=10)

  
    p.add_argument("--coarse_patch_size", type=int, default=128)
    p.add_argument("--coarse_stride", type=int, default=64)
    p.add_argument("--coarse_threshold", type=float, default=0.50)
    p.add_argument("--bbox_margin_mm", type=float, default=12.0)
    p.add_argument("--fallback_half_size_mm", type=float, default=30.0)
    p.add_argument("--hu_min", type=float, default=-1000.0)
    p.add_argument("--hu_max", type=float, default=1000.0)
    p.add_argument("--body_hu", type=float, default=-900.0)
    p.add_argument("--rebuild_prompt_cache", action="store_true")

    p.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow_partial_weight", action="store_true")
    p.add_argument("--port", type=int, default=12361)
    p.add_argument("--seed", type=int, default=2023)
    return p.parse_args()


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in args.gpu_ids)
MODEL_SAVE_PATH = osp.join(args.work_dir, args.task_name)
LOG_OUT_DIR = MODEL_SAVE_PATH
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
PROMPT_CACHE_PATH = osp.join(MODEL_SAVE_PATH, "coarse_autoprompt_cache.json")
logger = logging.getLogger(__name__)



def normalize_organ_name(name):
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    aliases = {
        "left_lens": "l_lens", "right_lens": "r_lens",
        "left_cochlea": "l_cochlea", "right_cochlea": "r_cochlea",
        "left_opticnerve": "l_opticnerve", "right_opticnerve": "r_opticnerve",
        "left_optic_nerve": "l_opticnerve", "right_optic_nerve": "r_opticnerve",
        "l_optic_nerve": "l_opticnerve", "r_optic_nerve": "r_opticnerve",
        "oral_cavity": "oralcavity", "spinal_cord": "spinalcord",
        "l_brachial_plexus": "l_brachialplexus", "r_brachial_plexus": "r_brachialplexus",
    }
    return aliases.get(s, s)


def strip_nii(name):
    name = str(name)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def case_id_from_name(name):
    cid = strip_nii(Path(name).name)
    if cid.endswith("_0000"):
        cid = cid[:-5]
    return cid


def load_nii_zyx(path):
    nii = nib.load(str(path))
    arr = np.asarray(nii.dataobj, dtype=np.float32)
    arr = np.transpose(arr, (2, 1, 0))
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3][::-1])
    return arr, spacing


def discover_samples(data_root):
    root = Path(data_root)
    samples = []
    for image_dir in root.rglob("imagesTr"):
        label_dir = image_dir.parent / "labelsTr"
        if not label_dir.exists():
            continue
        organ = normalize_organ_name(image_dir.relative_to(root).parts[0])
        labels = {case_id_from_name(p.name): p for p in list(label_dir.glob("*.nii.gz")) + list(label_dir.glob("*.nii"))}
        for img in sorted(list(image_dir.glob("*.nii.gz")) + list(image_dir.glob("*.nii"))):
            cid = case_id_from_name(img.name)
            if cid in labels:
                samples.append({
                    "case_id": cid,
                    "organ": organ,
                    "image": str(img),
                    "label": str(labels[cid]),
                })
    if not samples:
        raise RuntimeError(f"No SAM-FT-HN training samples found below {root}")
    return samples



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
        raise RuntimeError("Coarse checkpoint does not contain 'organs'.")
    state = ckpt.get("model_state_dict", ckpt)
    first_key = next((k for k in state if k.endswith("enc1.block.0.weight")), None)
    base = int(state[first_key].shape[0]) if first_key else 16
    model = CoarseUNet3D(1, len(organs), base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, [normalize_organ_name(o) for o in organs]



def ct_z_normalize_np(volume, hu_min, hu_max, body_hu):
    x = np.clip(volume.astype(np.float32), hu_min, hu_max)
    body = x > body_hu
    vals = x[body] if body.any() else x.reshape(-1)
    return ((x - float(vals.mean())) / max(float(vals.std()), 1e-6)).astype(np.float32)


def sliding_starts(length, patch, stride):
    if length <= patch:
        return [0]
    s = list(range(0, length-patch+1, stride))
    if s[-1] != length-patch:
        s.append(length-patch)
    return s


def pad_to_patch(x, patch):
    D, H, W = x.shape[-3:]
    pd, ph, pw = max(0, patch-D), max(0, patch-H), max(0, patch-W)
    return F.pad(x, (0,pw,0,ph,0,pd), value=0), (D,H,W)


@torch.no_grad()
def coarse_predict_all(model, image_np, patch, stride, device):
    x = torch.from_numpy(image_np)[None, None].float()
    x, original = pad_to_patch(x, patch)
    D,H,W = x.shape[-3:]
    C = model.out_conv.out_channels
    prob_sum = torch.zeros((C,D,H,W), dtype=torch.float32)
    count = torch.zeros((1,D,H,W), dtype=torch.float32)
    for z in sliding_starts(D,patch,stride):
        for y in sliding_starts(H,patch,stride):
            for xx in sliding_starts(W,patch,stride):
                p = x[..., z:z+patch, y:y+patch, xx:xx+patch].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    q = torch.sigmoid(model(p))[0].float().cpu()
                prob_sum[:, z:z+patch, y:y+patch, xx:xx+patch] += q
                count[:, z:z+patch, y:y+patch, xx:xx+patch] += 1
    prob_sum /= count.clamp_min(1)
    d,h,w = original
    return prob_sum[:, :d, :h, :w].numpy()


def largest_component(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == counts.argmax()


def probability_centroid(prob):
    p = np.asarray(prob, dtype=np.float64)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return tuple(int(v) for v in np.unravel_index(np.argmax(p), p.shape))
    coords = np.indices(p.shape)
    return tuple(int(np.clip(round((coords[i]*p).sum()/total), 0, p.shape[i]-1)) for i in range(3))


def mask_centroid(mask):
    pts = np.argwhere(mask)
    return None if pts.size == 0 else tuple(np.round(pts.mean(axis=0)).astype(int))


def mm_to_vox(mm, spacing):
    return tuple(max(1, int(round(float(mm)/max(float(s),1e-6)))) for s in spacing)


def auto_prompt_from_prob(prob, spacing, threshold, bbox_margin_mm, fallback_half_size_mm):
    comp = largest_component(prob >= threshold)
    center = mask_centroid(comp)
    if center is None:
        center = probability_centroid(prob)

    pts = np.argwhere(comp)
    if pts.size:
        margin = np.asarray(mm_to_vox(bbox_margin_mm, spacing))
        lo = np.maximum(pts.min(axis=0)-margin, 0)
        hi = np.minimum(pts.max(axis=0)+margin+1, np.asarray(prob.shape))
    else:
        half = np.asarray(mm_to_vox(fallback_half_size_mm, spacing))
        c = np.asarray(center)
        lo = np.maximum(c-half, 0)
        hi = np.minimum(c+half+1, np.asarray(prob.shape))
    return tuple(int(v) for v in center), tuple(int(v) for v in (*lo, *hi))



def build_prompt_cache(samples, cache_path, coarse_checkpoint, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coarse_model, coarse_organs = load_coarse_model(coarse_checkpoint, device)
    organ_to_idx = {o:i for i,o in enumerate(coarse_organs)}

    by_case = defaultdict(list)
    for s in samples:
        by_case[s["case_id"]].append(s)

    cache = {}
    for cid, case_samples in tqdm(sorted(by_case.items()), desc="Precomputing automatic coarse prompts"):
        image_path = case_samples[0]["image"]
        image, spacing = load_nii_zyx(image_path)
        image_norm = ct_z_normalize_np(image, args.hu_min, args.hu_max, args.body_hu)
        probs = coarse_predict_all(
            coarse_model, image_norm,
            args.coarse_patch_size, args.coarse_stride, device,
        )

        cache[cid] = {}
        for s in case_samples:
            organ = s["organ"]
            if organ not in organ_to_idx:
                continue
            center, box = auto_prompt_from_prob(
                probs[organ_to_idx[organ]], spacing,
                args.coarse_threshold, args.bbox_margin_mm,
                args.fallback_half_size_mm,
            )
            cache[cid][organ] = {
                "center_zyx": list(center),
                "bbox_zyxzyx": list(box),
                "image_path": image_path,
            }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

   
    del coarse_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache



def crop_with_info(arr, center, size, fill=0.0):
    spatial = np.asarray(arr.shape[-3:], dtype=int)
    center = np.asarray(center, dtype=int)
    start = center - size//2
    end = start + size
    src0, src1 = np.maximum(start,0), np.minimum(end,spatial)
    pad0, pad1 = np.maximum(-start,0), np.maximum(end-spatial,0)
    cropped = arr[(..., slice(src0[0],src1[0]), slice(src0[1],src1[1]), slice(src0[2],src1[2]))]
    pads = [(0,0)]*(arr.ndim-3) + [(int(pad0[0]),int(pad1[0])),(int(pad0[1]),int(pad1[1])),(int(pad0[2]),int(pad1[2]))]
    cropped = np.pad(cropped, pads, mode="constant", constant_values=fill)
    return cropped, {"start": start, "size": size}


def global_to_local(point, info):
    q = np.asarray(point, dtype=int) - np.asarray(info["start"], dtype=int)
    q = np.clip(q, 0, info["size"]-1)
    return tuple(int(v) for v in q)


def global_box_to_local(box, info):
    z0,y0,x0,z1,y1,x1 = box
    a = global_to_local((z0,y0,x0), info)
    b = global_to_local((z1-1,y1-1,x1-1), info)
    return tuple(int(v) for v in (*a,*b))


class SAMFTHNAutoPromptDataset(Dataset):
    def __init__(self, samples, prompt_cache, img_size, hu_min, hu_max, body_hu, intensity_aug=True):
        self.samples = [s for s in samples if s["case_id"] in prompt_cache and s["organ"] in prompt_cache[s["case_id"]]]
        self.cache = prompt_cache
        self.img_size = int(img_size)
        self.hu_min, self.hu_max, self.body_hu = hu_min, hu_max, body_hu
        self.intensity_aug = intensity_aug
        if not self.samples:
            raise RuntimeError("No training samples have a cached coarse prompt.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = self.cache[s["case_id"]][s["organ"]]
        center = tuple(prompt["center_zyx"])
        box = tuple(prompt["bbox_zyxzyx"])

        image, _ = load_nii_zyx(s["image"])
        gt, _ = load_nii_zyx(s["label"])
        image = ct_z_normalize_np(image, self.hu_min, self.hu_max, self.body_hu)
        gt = (gt > 0).astype(np.float32)

        image_crop, info = crop_with_info(image, center, self.img_size, fill=0.0)
        gt_crop, _ = crop_with_info(gt, center, self.img_size, fill=0.0)
        point_local = global_to_local(center, info)
        box_local = global_box_to_local(box, info)

        image_t = torch.from_numpy(image_crop)[None].float()
        gt_t = torch.from_numpy(gt_crop)[None].float()

        
        if self.intensity_aug and random.random() < 0.5:
            image_t = image_t * random.uniform(0.95, 1.05) + random.uniform(-0.10, 0.10)
            ns = random.uniform(0.0, 0.03)
            if ns > 0:
                image_t = image_t + torch.randn_like(image_t) * ns

        return {
            "image": image_t,
            "label": gt_t,
            "point": torch.tensor(point_local, dtype=torch.float32),
            "box": torch.tensor(box_local, dtype=torch.float32),
            "organ": s["organ"],
            "case_id": s["case_id"],
        }



def encode_auto_prompt(prompt_encoder, points, labels, boxes, low_res_masks):
   
    candidates = [boxes, boxes.view(boxes.shape[0], 2, 3)]
    for b in candidates:
        try:
            return prompt_encoder(points=[points, labels], boxes=b, masks=low_res_masks)
        except (TypeError, RuntimeError, ValueError, AssertionError):
            pass
    return prompt_encoder(points=[points, labels], boxes=None, masks=low_res_masks)




class BaseTrainer:
    def __init__(self, model, loader, args):
        self.model = model
        self.loader = loader
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.losses, self.dices = [], []
        self.writer = SummaryWriter(log_dir=osp.join(LOG_OUT_DIR, "tensorboard"))
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.set_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.step_size, gamma=args.gamma
        )
        self.start_epoch = 0
        self.init_checkpoint(
            osp.join(args.work_dir, args.task_name, "sam_model_latest.pth") if args.resume else args.checkpoint
        )

    @property
    def sam(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def set_optimizer(self):
        sam = self.model.module if isinstance(self.model, DDP) else self.model
        self.optimizer = torch.optim.AdamW([
            {"params": sam.image_encoder.parameters(), "lr": self.args.lr},
            {"params": sam.prompt_encoder.parameters(), "lr": self.args.lr * 0.1},
            {"params": sam.mask_decoder.parameters(), "lr": self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def init_checkpoint(self, path):
        if not path or not osp.exists(path):
            print(f"No checkpoint found at {path}; training from scratch")
            return
        ckpt = torch.load(path, map_location=self.args.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.sam.load_state_dict(state, strict=not self.args.allow_partial_weight)
        if self.args.resume and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            self.start_epoch = int(ckpt.get("epoch", 0))
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "lr_scheduler_state_dict" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            self.losses = ckpt.get("losses", [])
            self.dices = ckpt.get("dices", [])
            self.best_loss = ckpt.get("best_loss", np.inf)
            self.best_dice = ckpt.get("best_dice", 0.0)
        print(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, epoch, describe):
        if getattr(self.args, "rank", 0) not in (-1, 0):
            return
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.sam.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "training_prompt_source": "frozen_coarse_localizer",
            "prompt_cache": PROMPT_CACHE_PATH,
            "args": vars(self.args),
        }, osp.join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    @staticmethod
    def dice_score(logits, gt):
        pred = torch.sigmoid(logits) > 0.5
        gt = gt > 0
        inter = (pred & gt).sum(dim=(1,2,3,4)).float()
        den = pred.sum(dim=(1,2,3,4)).float() + gt.sum(dim=(1,2,3,4)).float()
        valid = den > 0
        if not valid.any():
            return float("nan")
        return float((2*inter[valid]/den[valid]).mean().item())

    def forward_batch(self, batch):
        image = batch["image"].to(self.args.device, non_blocking=True)
        gt = batch["label"].to(self.args.device, non_blocking=True)
        point = batch["point"].to(self.args.device, non_blocking=True).unsqueeze(1)  # B,1,3
        labels = torch.ones((point.shape[0],1), dtype=torch.float32, device=self.args.device)
        boxes = batch["box"].to(self.args.device, non_blocking=True)  # B,6

        sam = self.sam
        image_embedding = sam.image_encoder(image)
        low_shape = (self.args.img_size//4,)*3
        low_prompt = torch.zeros((image.shape[0],1,*low_shape), device=self.args.device)
        sparse, dense = encode_auto_prompt(sam.prompt_encoder, point, labels, boxes, low_prompt)
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(low_res_masks, size=gt.shape[-3:], mode="trilinear", align_corners=False)
        loss = self.seg_loss(logits, gt.float())
        return logits, gt, loss

    def train_epoch(self, epoch):
        self.model.train()
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(epoch)

        total_loss, dice_values = 0.0, []
        self.optimizer.zero_grad(set_to_none=True)
        scaler = self.scaler
        tbar = tqdm(self.loader, disable=getattr(self.args,"rank",0) not in (-1,0))

        for step, batch in enumerate(tbar):
            sync_step = ((step + 1) % self.args.accumulation_steps == 0) or (step + 1 == len(self.loader))
            context = nullcontext()
            if isinstance(self.model, DDP) and not sync_step:
                context = self.model.no_sync()

            with context:
                with amp.autocast(enabled=torch.cuda.is_available()):
                    logits, gt, raw_loss = self.forward_batch(batch)
                    loss = raw_loss / self.args.accumulation_steps
                scaler.scale(loss).backward()

            total_loss += raw_loss.item()
            dice_values.append(self.dice_score(logits.detach(), gt))

            if sync_step:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if getattr(self.args,"rank",0) in (-1,0):
                tbar.set_postfix(loss=f"{raw_loss.item():.4f}")

        finite_dice = [d for d in dice_values if np.isfinite(d)]
        return total_loss/max(1,len(self.loader)), float(np.mean(finite_dice)) if finite_dice else float("nan")

    def train(self):
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())
        for epoch in range(self.start_epoch, self.args.num_epochs):
            loss, dice = self.train_epoch(epoch)
            self.lr_scheduler.step()

            if getattr(self.args,"rank",0) in (-1,0):
                self.losses.append(loss); self.dices.append(dice)
                self.writer.add_scalar("Loss/train", loss, epoch)
                self.writer.add_scalar("Dice/train", dice, epoch)
                print(f"Epoch {epoch+1}/{self.args.num_epochs}: loss={loss:.5f}, Dice={dice:.5f}")

                self.save_checkpoint(epoch, "latest")
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(epoch, "loss_best")
                if np.isfinite(dice) and dice > self.best_dice:
                    self.best_dice = dice
                    self.save_checkpoint(epoch, "dice_best")
                if self.args.save_every > 0 and ((epoch+1) % self.args.save_every == 0 or epoch+1 == self.args.num_epochs):
                    self.save_checkpoint(epoch, f"epoch_{epoch+1:03d}")

        if getattr(self.args,"rank",0) in (-1,0):
            self.writer.close()



def init_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def setup(rank, world_size, port):
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size, rank=rank,
    )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_loader(samples, cache, args, rank=None, world_size=None):
    ds = SAMFTHNAutoPromptDataset(
        samples, cache, args.img_size,
        args.hu_min, args.hu_max, args.body_hu,
        intensity_aug=True,
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if rank is not None else None
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0,
    )


def worker(rank, world_size, samples, cache, args):
    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)
    args.rank = rank
    args.device = torch.device(f"cuda:{rank}")
    init_seeds(args.seed + rank)

    loader = build_loader(samples, cache, args, rank, world_size)
    model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    trainer = BaseTrainer(model, loader, args)
    trainer.train()
    cleanup()


def main():
    mp.set_sharing_strategy("file_system")
    init_seeds(args.seed)
    samples = discover_samples(args.data_root)
    print(f"Found {len(samples)} organ-specific training samples")

    if args.rebuild_prompt_cache or not osp.exists(PROMPT_CACHE_PATH):
        cache = build_prompt_cache(samples, PROMPT_CACHE_PATH, args.coarse_checkpoint, args)
    else:
        with open(PROMPT_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded automatic prompt cache: {PROMPT_CACHE_PATH}")

    if args.multi_gpu:
        world_size = len(args.gpu_ids)
        mp.spawn(worker, nprocs=world_size, args=(world_size, samples, cache, args))
    else:
        args.rank = -1
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader = build_loader(samples, cache, args)
        model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
        trainer = BaseTrainer(model, loader, args)
        trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

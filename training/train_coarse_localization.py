import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_SMALL_ORGANS = {
    "l_lens", "r_lens", "chiasm",
    "l_cochlea", "r_cochlea",
    "l_opticnerve", "r_opticnerve",
}


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
        "l_brachial_plexus": "l_brachialplexus",
        "r_brachial_plexus": "r_brachialplexus",
    }
    return aliases.get(s, s)

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
        self.enc1 = DoubleConv3D(in_channels, b)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(b, b * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(b * 2, b * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(b * 4, b * 8)
        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv3D(b * 2, b)
        self.out_conv = nn.Conv3d(b, out_channels, 1)

    @staticmethod
    def _match(x, ref):
        if x.shape[-3:] != ref.shape[-3:]:
            x = F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.out_conv(d1)


def strip_nii_suffix(name):
    name = str(name)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def normalize_case_id(filename):
    cid = strip_nii_suffix(Path(filename).name)
    if cid.endswith("_0000"):
        cid = cid[:-5]
    return cid


def load_nifti_zyx(path):
    nii = nib.load(str(path))
    arr = np.asarray(nii.dataobj, dtype=np.float32)
    arr = np.transpose(arr, (2, 1, 0))  # X,Y,Z -> Z,Y,X
    spacing_xyz = nii.header.get_zooms()[:3]
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    return arr, spacing_zyx


def discover_dataset(data_root):
    data_root = Path(data_root).resolve()
    image_dirs = list(data_root.rglob("imagesTr"))
    if not image_dirs:
        raise RuntimeError(f"No imagesTr folders found under {data_root}")

    cases = defaultdict(lambda: {"image": None, "masks": {}})
    organ_names = set()

    for image_dir in image_dirs:
        label_dir = image_dir.parent / "labelsTr"
        if not label_dir.exists():
            continue
        organ = normalize_organ_name(image_dir.relative_to(data_root).parts[0])
        organ_names.add(organ)
        images = sorted(list(image_dir.glob("*.nii.gz")) + list(image_dir.glob("*.nii")))
        labels = sorted(list(label_dir.glob("*.nii.gz")) + list(label_dir.glob("*.nii")))
        label_lookup = {normalize_case_id(p.name): p for p in labels}
        for img in images:
            cid = normalize_case_id(img.name)
            if cases[cid]["image"] is None:
                cases[cid]["image"] = img
            if cid in label_lookup:
                cases[cid]["masks"][organ] = label_lookup[cid]

    cases = {cid: v for cid, v in cases.items() if v["image"] is not None and v["masks"]}
    organs = sorted(organ_names)
    if not cases:
        raise RuntimeError("No matching CT/mask pairs were found.")
    return cases, organs


def ct_z_normalize_np(volume, hu_min=-1000.0, hu_max=1000.0, body_hu=-900.0):
    x = np.clip(volume.astype(np.float32), hu_min, hu_max)
    body = x > body_hu
    vals = x[body] if body.any() else x.reshape(-1)
    mean = float(vals.mean())
    std = max(float(vals.std()), 1e-6)
    return ((x - mean) / std).astype(np.float32)


def fixed_crop_np(arr, center, patch_size, fill=0.0):
    spatial = np.array(arr.shape[-3:], dtype=int)
    size = np.array([patch_size] * 3, dtype=int)
    center = np.asarray(center, dtype=int)
    start = center - size // 2
    end = start + size
    src0 = np.maximum(start, 0)
    src1 = np.minimum(end, spatial)
    pad0 = np.maximum(-start, 0)
    pad1 = np.maximum(end - spatial, 0)

    cropped = arr[(..., slice(src0[0], src1[0]), slice(src0[1], src1[1]), slice(src0[2], src1[2]))]
    pad_width = [(0, 0)] * (arr.ndim - 3) + [
        (int(pad0[0]), int(pad1[0])),
        (int(pad0[1]), int(pad1[1])),
        (int(pad0[2]), int(pad1[2])),
    ]
    return np.pad(cropped, pad_width, mode="constant", constant_values=fill)


def random_center(shape):
    return tuple(random.randrange(max(1, int(s))) for s in shape)


def foreground_center(mask):
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return None
    p = pts[random.randrange(len(pts))]
    return tuple(int(v) for v in p)


def augment_patch(image, masks):

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image),
        label=tio.LabelMap(tensor=masks),
    )
    if random.random() < 0.6:
        subject = tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=10,
            translation=0,
            image_interpolation="linear",
            label_interpolation="nearest",
            default_pad_value="minimum",
        )(subject)

    image = subject.image.data.float()
    masks = (subject.label.data > 0.5).float()

    if random.random() < 0.5:
        image = image * random.uniform(0.95, 1.05) + random.uniform(-0.10, 0.10)
        noise_std = random.uniform(0.0, 0.03)
        if noise_std > 0:
            image = image + torch.randn_like(image) * noise_std
    return image, masks


class CoarsePatchDataset(Dataset):
    def __init__(self, case_ids, case_dict, organs, patch_size=128, patches_per_case=8,
                 organ_center_prob=0.5, small_organ_weight=3.0, small_organs=None,
                 hu_min=-1000.0, hu_max=1000.0, body_hu=-900.0, augment=True):
        self.case_ids = list(case_ids)
        self.case_dict = case_dict
        self.organs = list(organs)
        self.patch_size = int(patch_size)
        self.patches_per_case = int(patches_per_case)
        self.organ_center_prob = float(organ_center_prob)
        self.small_organ_weight = float(small_organ_weight)
        self.small_organs = set(small_organs or DEFAULT_SMALL_ORGANS)
        self.hu_min, self.hu_max, self.body_hu = hu_min, hu_max, body_hu
        self.augment = augment
        self._cache_id = None
        self._cache = None

    def __len__(self):
        return max(1, len(self.case_ids) * self.patches_per_case)

    def _load_case(self, cid):
        if self._cache_id == cid:
            return self._cache
        info = self.case_dict[cid]
        image, _ = load_nifti_zyx(info["image"])
        image = ct_z_normalize_np(image, self.hu_min, self.hu_max, self.body_hu)
        masks, available = [], []
        for organ in self.organs:
            p = info["masks"].get(organ)
            if p is None:
                masks.append(np.zeros_like(image, dtype=np.float32))
                available.append(0.0)
            else:
                m, _ = load_nifti_zyx(p)
                masks.append((m > 0).astype(np.float32))
                available.append(1.0)
        result = (image, np.stack(masks, axis=0), np.asarray(available, dtype=np.float32))
        self._cache_id, self._cache = cid, result
        return result

    def _choose_center(self, masks, available):
        if random.random() >= self.organ_center_prob:
            return random_center(masks.shape[-3:])
        candidates, weights = [], []
        for c, organ in enumerate(self.organs):
            if available[c] > 0.5 and masks[c].any():
                candidates.append(c)
                weights.append(self.small_organ_weight if organ in self.small_organs else 1.0)
        if not candidates:
            return random_center(masks.shape[-3:])
        c = random.choices(candidates, weights=weights, k=1)[0]
        return foreground_center(masks[c]) or random_center(masks.shape[-3:])

    def __getitem__(self, idx):
        cid = self.case_ids[idx % len(self.case_ids)]
        image, masks, available = self._load_case(cid)
        center = self._choose_center(masks, available)
        image = torch.from_numpy(fixed_crop_np(image, center, self.patch_size)[None]).float()
        masks = torch.from_numpy(fixed_crop_np(masks, center, self.patch_size)).float()
        available = torch.from_numpy(available).float()
        if self.augment:
            image, masks = augment_patch(image, masks)
        return {"image": image, "mask": masks, "available": available, "case_id": cid}



def masked_soft_dice_loss(logits, targets, available, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3, 4))
    den = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - (dice * available).sum() / available.sum().clamp_min(1.0)


def masked_focal_loss(logits, targets, available, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - pt).pow(gamma) * bce
    focal = focal.mean(dim=(2, 3, 4))
    return (focal * available).sum() / available.sum().clamp_min(1.0)


def combined_loss(logits, targets, available, dice_weight=1.0, focal_weight=1.0,
                  gamma=2.0, alpha=0.25):
    dice = masked_soft_dice_loss(logits, targets, available)
    focal = masked_focal_loss(logits, targets, available, gamma, alpha)
    return dice_weight * dice + focal_weight * focal, dice, focal


def sliding_starts(length, patch, stride):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def pad_to_patch(x, patch_size):
    D, H, W = x.shape[-3:]
    pd, ph, pw = max(0, patch_size-D), max(0, patch_size-H), max(0, patch_size-W)
    return F.pad(x, (0, pw, 0, ph, 0, pd), value=0), (D, H, W)


@torch.no_grad()
def sliding_window_predict_all(model, image, patch_size, stride, device, amp_enabled=True):
    model.eval()
    image, original_shape = pad_to_patch(image, patch_size)
    D, H, W = image.shape[-3:]
    zs = sliding_starts(D, patch_size, stride)
    ys = sliding_starts(H, patch_size, stride)
    xs = sliding_starts(W, patch_size, stride)

    C = model.out_conv.out_channels
    prob_sum = torch.zeros((C, D, H, W), dtype=torch.float32)
    count = torch.zeros((1, D, H, W), dtype=torch.float32)

    for z in zs:
        for y in ys:
            for x in xs:
                patch = image[..., z:z+patch_size, y:y+patch_size, x:x+patch_size].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(amp_enabled and device.type == "cuda")):
                    prob = torch.sigmoid(model(patch))[0].float().cpu()
                prob_sum[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += prob
                count[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1

    prob_sum /= count.clamp_min(1.0)
    d, h, w = original_shape
    return prob_sum[:, :d, :h, :w]


def largest_component(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def binary_dice(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    den = pred.sum() + gt.sum()
    if den == 0:
        return float("nan")
    return float((2 * np.logical_and(pred, gt).sum() + eps) / (den + eps))


def centroid(mask):
    pts = np.argwhere(mask)
    return None if pts.size == 0 else pts.mean(axis=0)


def centroid_error_mm(pred, gt, spacing):
    cp, cg = centroid(pred), centroid(gt)
    if cp is None or cg is None:
        return float("nan")
    return float(np.linalg.norm((cp - cg) * np.asarray(spacing)))


def roi_coverage(pred, gt, spacing, margin_mm=12.0):
    if gt.sum() == 0:
        return float("nan")
    pts = np.argwhere(pred)
    if pts.size == 0:
        return 0.0
    margin = np.asarray([max(1, int(round(margin_mm / max(s, 1e-6)))) for s in spacing])
    lo = np.maximum(pts.min(axis=0) - margin, 0)
    hi = np.minimum(pts.max(axis=0) + margin + 1, np.asarray(gt.shape))
    roi = np.zeros_like(gt, dtype=bool)
    roi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return float(np.logical_and(roi, gt).sum() / gt.sum())


@torch.no_grad()
def evaluate_full_volumes(model, case_ids, case_dict, organs, device, args):
    metrics = defaultdict(lambda: {"dice": [], "centroid_error_mm": [], "roi_coverage": [], "empty": []})

    for cid in tqdm(case_ids, desc="Full-volume validation", leave=False):
        info = case_dict[cid]
        image_np, spacing = load_nifti_zyx(info["image"])
        image_np = ct_z_normalize_np(image_np, args.hu_min, args.hu_max, args.body_hu)
        probs = sliding_window_predict_all(
            model, torch.from_numpy(image_np)[None, None].float(),
            args.patch_size, args.stride, device, args.amp,
        ).numpy()

        for c, organ in enumerate(organs):
            mask_path = info["masks"].get(organ)
            if mask_path is None:
                continue
            gt, _ = load_nifti_zyx(mask_path)
            gt = gt > 0
            pred = largest_component(probs[c] >= args.coarse_threshold)
            metrics[organ]["dice"].append(binary_dice(pred, gt))
            metrics[organ]["centroid_error_mm"].append(centroid_error_mm(pred, gt, spacing))
            metrics[organ]["roi_coverage"].append(roi_coverage(pred, gt, spacing, args.roi_margin_mm))
            metrics[organ]["empty"].append(float(not pred.any()))

    def meanfinite(values):
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")

    per_organ = {}
    all_dice, all_ce, all_cov, all_empty = [], [], [], []
    for organ in organs:
        m = metrics[organ]
        per_organ[organ] = {
            "dice": meanfinite(m["dice"]),
            "centroid_error_mm": meanfinite(m["centroid_error_mm"]),
            "roi_coverage": meanfinite(m["roi_coverage"]),
            "empty_rate": meanfinite(m["empty"]),
        }
        all_dice += m["dice"]
        all_ce += m["centroid_error_mm"]
        all_cov += m["roi_coverage"]
        all_empty += m["empty"]

    return {
        "mean_dice": meanfinite(all_dice),
        "mean_centroid_error_mm": meanfinite(all_ce),
        "mean_roi_coverage": meanfinite(all_cov),
        "empty_rate": meanfinite(all_empty),
        "per_organ": per_organ,
    }


def train(args):
    seed_everything(args.seed)
    data_root = Path(args.data_root)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    case_dict, organs = discover_dataset(data_root)
    small_organs = {normalize_organ_name(x) for x in args.small_organs.split(",") if x.strip()}

    case_ids = sorted(case_dict)
    random.Random(args.seed).shuffle(case_ids)
    n_val = max(1, int(round(len(case_ids) * args.val_fraction)))
    val_ids, train_ids = case_ids[:n_val], case_ids[n_val:]

    with open(ckpt_dir / "dataset_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "organs": organs,
            "train_cases": train_ids,
            "val_cases": val_ids,
            "small_organs": sorted(small_organs),
        }, f, indent=2)

    logging.info("Patients: total=%d train=%d val=%d", len(case_ids), len(train_ids), len(val_ids))
    logging.info("Organs (%d): %s", len(organs), organs)
    logging.info("Coarse patches: %d^3, stride=%d", args.patch_size, args.stride)

    train_ds = CoarsePatchDataset(
        train_ids, case_dict, organs,
        patch_size=args.patch_size,
        patches_per_case=args.patches_per_case,
        organ_center_prob=args.organ_center_prob,
        small_organ_weight=args.small_organ_weight,
        small_organs=small_organs,
        hu_min=args.hu_min, hu_max=args.hu_max, body_hu=args.body_hu,
        augment=True,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CoarseUNet3D(1, len(organs), args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(3, dtype=float)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            available = batch["available"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(image)
                loss, dice_l, focal_l = combined_loss(
                    logits, target, available,
                    args.dice_weight, args.focal_weight,
                    args.focal_gamma, args.focal_alpha,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += [loss.item(), dice_l.item(), focal_l.item()]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        running /= max(1, len(loader))
        do_val = epoch % args.validate_every == 0 or epoch == args.epochs
        val_metrics = None
        val_dice = None
        if do_val:
            val_metrics = evaluate_full_volumes(model, val_ids, case_dict, organs, device, args)
            val_dice = val_metrics["mean_dice"]
            scheduler.step(val_dice)
            logging.info(
                "Epoch %03d | train %.4f (DiceLoss %.4f Focal %.4f) | "
                "val Dice %.4f | centroid %.3f mm | ROI %.4f | empty %.4f",
                epoch, running[0], running[1], running[2], val_dice,
                val_metrics["mean_centroid_error_mm"],
                val_metrics["mean_roi_coverage"], val_metrics["empty_rate"],
            )
        else:
            logging.info("Epoch %03d | train %.4f (DiceLoss %.4f Focal %.4f)",
                         epoch, running[0], running[1], running[2])

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "organs": organs,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "coarse_threshold": args.coarse_threshold,
            "normalization": {
                "type": "ct_clip_body_zscore",
                "hu_min": args.hu_min,
                "hu_max": args.hu_max,
                "body_hu": args.body_hu,
            },
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(state, ckpt_dir / "coarse_last.pth")
        if do_val and np.isfinite(val_dice) and val_dice > best_val:
            best_val = float(val_dice)
            torch.save(state, ckpt_dir / "coarse_best.pth")
            with open(ckpt_dir / "coarse_best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
            logging.info("New best coarse checkpoint: %.4f", best_val)

    logging.info("Training complete. Best validation Dice: %.4f", best_val)


def get_args():
    p = argparse.ArgumentParser(description="Train SAM-FT-HN coarse localizer")
    p.add_argument("--data-root", default=r"H:\Research\SAM-FT-HN\medical_preprocessed")
    p.add_argument("--checkpoint-dir", default=r"H:\Research\SAM-FT-HN\coarse_checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--patches-per-case", type=int, default=8)
    p.add_argument("--organ-center-prob", type=float, default=0.50)
    p.add_argument("--small-organ-weight", type=float, default=3.0)
    p.add_argument("--small-organs", default="l_lens,r_lens,chiasm,l_cochlea,r_cochlea,l_opticnerve,r_opticnerve")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--focal-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--hu-min", type=float, default=-1000.0)
    p.add_argument("--hu-max", type=float, default=1000.0)
    p.add_argument("--body-hu", type=float, default=-900.0)
    p.add_argument("--coarse-threshold", type=float, default=0.50)
    p.add_argument("--roi-margin-mm", type=float, default=12.0)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=12.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train(get_args())

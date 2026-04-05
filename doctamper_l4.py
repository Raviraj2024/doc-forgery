#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DocTamper — LOCAL L4 GPU TRAINING  |  Windows-Compatible                  ║
# ║  Techniques: ELA · SRM · Noiseprint · DINO-ViT · OCR-proxy · P-Hash        ║
# ║  F1 Target: >80%  |  Output: ./output/forgery_best.pth                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, io, gc, sys, json, random, time, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from pathlib import Path
from PIL import Image
import lmdb
import imagehash
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  — adjust DATA_ROOT if your unzipped folder is elsewhere
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT  = Path("./dinmkeljiame/doctamper/versions/1")
OUTPUT_DIR = Path("./output")
CACHE_DIR  = None

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_SAMPLES = 20_000
VAL_SAMPLES   = 3000
EPOCHS        = 25
BATCH         = 8
GRAD_ACC      = 2
IMG_SIZE      = 512
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7
N_CH          = 13
ENCODER       = "mit_b4"

# ══════════════════════════════════════════════════════════════════════════════
#  CPU FEATURE FUNCTIONS  (defined at module level so workers can pickle them)
# ══════════════════════════════════════════════════════════════════════════════

# FIND the entire compute_ela_multi function and REPLACE with:
def compute_ela_multi(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, "JPEG", quality=90)
    buf.seek(0)
    comp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    orig = np.array(img_pil, dtype=np.float32)
    diff = np.abs(orig - comp)
    return (diff * 255.0 / (diff.max() + 1e-6)).astype(np.uint8)


def compute_laplacian(img_np):
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fine   = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    blur3  = cv2.GaussianBlur(gray, (3, 3), 0)
    medium = np.abs(cv2.Laplacian(blur3, cv2.CV_32F, ksize=3))
    blur5  = cv2.GaussianBlur(gray, (5, 5), 0)
    coarse = np.abs(cv2.Laplacian(blur5, cv2.CV_32F, ksize=5))
    lap    = np.stack([fine, medium, coarse], axis=2)
    return (lap / (lap.max() + 1e-6) * 255).astype(np.uint8)


def compute_ocr_proxy(img_np):
    gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kern     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad     = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kern)
    h_kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    text_map = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, h_kern)
    return text_map


# RGB-only augmentation (must be at module level for pickling on Windows)
_rgb_aug = A.Compose([
    A.CLAHE(clip_limit=3.0, p=0.35),
    A.Sharpen(alpha=(0.15, 0.4), p=0.35),
    A.ColorJitter(brightness=0.1, contrast=0.1,
                  saturation=0.1, hue=0.05, p=0.3),
])

_spatial_train = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.35),
    A.Rotate(limit=20, p=0.4),
    A.ElasticTransform(alpha=1, sigma=50, p=0.25),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),
    A.CoarseDropout(num_holes_range=(1, 8),
                    hole_height_range=(8, 48),
                    hole_width_range=(8, 48), p=0.25),
    A.RandomScale(scale_limit=0.15, p=0.3),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE,
                  border_mode=cv2.BORDER_REFLECT),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    ToTensorV2(),
])

_spatial_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2(),
])


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (module level — required for Windows multiprocessing spawn)
# ══════════════════════════════════════════════════════════════════════════════

class DocTamperDataset(Dataset):
    def __init__(self, lmdb_path, indices, spatial_tf,
                 is_train=False, cache_dir=None):
        self.lmdb_path = str(lmdb_path)
        self.indices   = indices
        self.spatial_tf= spatial_tf
        self.is_train  = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.env       = None  # lazy open per worker

    def _open(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True,
                lock=False, meminit=False, readahead=False)

    def __len__(self):
        return len(self.indices)

    def _compute_features(self, img_pil, img_np):
        if self.is_train:
            aug    = _rgb_aug(image=img_np)
            img_np = aug["image"]
            img_pil= Image.fromarray(img_np)
        ela   = compute_ela_multi(img_pil)
        lap   = compute_laplacian(img_np)
        ocr   = compute_ocr_proxy(img_np)
        img_10_np = np.concatenate(
            [img_np, ela, lap, ocr[:, :, np.newaxis]], axis=2)
        ocr3   = np.stack([ocr, ocr, ocr], axis=2)
        img_12 = np.concatenate([img_np, ela, lap, ocr3], axis=2)
        return img_10_np, img_12

    def __getitem__(self, idx):
        self._open()
        i = self.indices[idx]

        with self.env.begin(write=False) as txn:
            img_b = txn.get(f"image-{i:09d}".encode())
            msk_b = txn.get(f"label-{i:09d}".encode())

        img_pil = Image.open(io.BytesIO(img_b)).convert("RGB")
        msk_pil = Image.open(io.BytesIO(msk_b)).convert("L")
        img_np  = np.array(img_pil)
        msk_np  = (np.array(msk_pil) > 127).astype(np.float32)

        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{i:09d}.npy"
            if cache_file.exists():
                img_10_np = np.load(str(cache_file))
                ocr3      = np.stack(
                    [img_10_np[:,:,9]]*3, axis=2)
                img_12    = np.concatenate(
                    [img_10_np[:,:,:9], ocr3], axis=2)
            else:
                img_10_np, img_12 = self._compute_features(img_pil, img_np)
                np.save(str(cache_file), img_10_np)
        else:
            _, img_12 = self._compute_features(img_pil, img_np)

        if self.spatial_tf:
            aug    = self.spatial_tf(image=img_12, mask=msk_np)
            img_12 = aug["image"]
            msk_np = aug["mask"]

        img_10 = torch.cat([img_12[:9], img_12[9:10]], dim=0)
        return img_10.float(), msk_np.unsqueeze(0).float()


# ══════════════════════════════════════════════════════════════════════════════
#  GPU EXTRACTORS  (module level)
# ══════════════════════════════════════════════════════════════════════════════

class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        k1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                        [0,-1,2,-1,0],[0,0,0,0,0]], np.float32) / 4.
        k2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                        [2,-6,8,-6,2],[-1,2,-2,2,-1]], np.float32) / 12.
        k3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
                        [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
        k4 = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],
                        [0,0,1,0,0],[0,0,0,0,0]], np.float32)
        k5 = np.array([[1,-2,1,0,0],[-2,4,-2,0,0],[1,-2,1,0,0],
                        [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 4.
        kernels = np.stack([k1,k2,k3,k4,k5], axis=0)[:, np.newaxis]
        self.register_buffer("kernels", torch.tensor(kernels))

    @torch.no_grad()
    def forward(self, rgb):
        gray = (0.299*rgb[:,0]+0.587*rgb[:,1]+0.114*rgb[:,2]).unsqueeze(1)/255.
        out  = F.conv2d(gray.float(), self.kernels.float(), padding=2)
        srm  = out.abs().mean(dim=1, keepdim=True)
        return srm / (srm.amax(dim=(2,3), keepdim=True) + 1e-6)


class NoiseprintExtractor(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        k   = 5
        x   = torch.arange(k).float() - k // 2
        g1d = torch.exp(-x**2 / (2*sigma**2))
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / g2d.sum()
        self.register_buffer("kernel", g2d.unsqueeze(0).unsqueeze(0))

    @torch.no_grad()
    def forward(self, rgb):
        gray     = (0.299*rgb[:,0]+0.587*rgb[:,1]+0.114*rgb[:,2]).unsqueeze(1)/255.
        smooth   = F.conv2d(gray.float(), self.kernel.float(), padding=2)
        residual = (gray - smooth).abs()
        return residual / (residual.amax(dim=(2,3), keepdim=True) + 1e-6)


class DinoViTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        loaded = False
        for name in ["vit_small_patch16_224", "vit_tiny_patch16_224"]:
            try:
                self.model = timm.create_model(
                    name, pretrained=True, num_classes=0)
                print(f"  ✓ DINO-ViT : {name}")
                loaded = True
                break
            except Exception as e:
                print(f"  ⚠ {name}: {e}")
        if not loaded:
            self.model = timm.create_model(
                "vit_tiny_patch16_224", pretrained=False, num_classes=0)
            print("  ⚠ DINO-ViT : random init")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.register_buffer("mean",
            torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",
            torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, rgb, out_h, out_w):
        x = rgb.float() / 255.0
        x = (x - self.mean) / (self.std + 1e-6)
        x = F.interpolate(x, (224,224), mode="bilinear", align_corners=False)
        x = x.to(next(self.model.parameters()).dtype)
        feats   = self.model.forward_features(x)
        patches = feats[:, 1:]
        cls     = feats[:, 0:1]
        dist    = torch.norm(patches - cls, dim=-1)
        mn, mx  = dist.amin(1, keepdim=True), dist.amax(1, keepdim=True)
        dist    = (dist - mn) / (mx - mn + 1e-6)
        n       = int(dist.shape[1] ** 0.5)
        dist    = dist.reshape(-1, 1, n, n)
        return F.interpolate(dist, (out_h, out_w),
                             mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.pool = nn.MaxPool2d(k, stride=1, padding=k//2)

    def forward(self, preds, masks):
        dilated  = self.pool(masks)
        eroded   = 1.0 - self.pool(1.0 - masks)
        boundary = (dilated - eroded).clamp(0, 1)
        loss = F.binary_cross_entropy_with_logits(
            preds * boundary, masks * boundary, reduction="sum"
        ) / (boundary.sum() + 1e-6)
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth= smooth

    def forward(self, preds, masks):
        p  = torch.sigmoid(preds)
        tp = (p * masks).sum()
        fp = (p * (1 - masks)).sum()
        fn = ((1 - p) * masks).sum()
        tv = (tp + self.smooth) / (
            tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tv


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def check_dataset(data_root):
    required = [
        data_root / "DocTamperV1-TrainingSet" / "data.mdb",
        data_root / "DocTamperV1-TestingSet"  / "data.mdb",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n" + "="*60)
        print("  DATASET NOT FOUND")
        print("="*60)
        for p in missing:
            print(f"  Missing: {p}")
        print("\n  Download steps:")
        print("  1. pip install kaggle")
        print("  2. Place kaggle.json in %USERPROFILE%\\.kaggle\\")
        print("  3. kaggle datasets download dinmkeljiame/doctamper")
        print("  4. Unzip into ./  (creates dinmkeljiame/doctamper/...)")
        sys.exit(1)
    print("✓ Dataset found")


def phash_deduplicate(lmdb_path, all_indices,
                      sample_n=10_000, threshold=8, seed=42):
    print(f"  [P-Hash] Checking {min(sample_n,len(all_indices))} images...")
    rng       = random.Random(seed)
    sampled   = rng.sample(all_indices, min(sample_n, len(all_indices)))
    unsampled = list(set(all_indices) - set(sampled))
    env = lmdb.open(str(lmdb_path), readonly=True,
                    lock=False, meminit=False, readahead=False)
    seen_hashes, unique_idx = [], []
    with env.begin() as txn:
        for idx in sampled:
            val = txn.get(f"image-{idx:09d}".encode())
            if val is None:
                continue
            img = Image.open(io.BytesIO(val)).convert("RGB").resize((64,64))
            h   = imagehash.phash(img, hash_size=8)
            if all(abs(h - s) > threshold for s in seen_hashes):
                seen_hashes.append(h)
                unique_idx.append(idx)
    env.close()
    removed = len(sampled) - len(unique_idx)
    kept    = unique_idx + unsampled
    print(f"  [P-Hash] Removed {removed} duplicates. "
          f"Using {len(kept):,} indices.")
    return kept


_CH_MEAN = torch.tensor([
    123.675, 116.28, 103.53,
    127.5, 127.5, 127.5,
    127.5, 127.5, 127.5,
    127.5, 127.5, 127.5, 127.5,
], dtype=torch.float32).view(1, N_CH, 1, 1)

_CH_STD = torch.tensor([
    58.395, 57.12, 57.375,
    64., 64., 64.,
    64., 64., 64.,
    64., 64., 64., 64.,
], dtype=torch.float32).view(1, N_CH, 1, 1)


def normalise(t):
    mean = _CH_MEAN.to(t.device)
    std  = _CH_STD.to(t.device)
    return (t - mean) / (std + 1e-6)


def gpu_features(imgs_10, device, srm_net, noiseprint, dino_vit):
    rgb    = imgs_10[:, 0:3]
    _, _, H, W = imgs_10.shape
    with torch.no_grad():
        srm_ch  = srm_net(rgb.half()).float() * 255.
        nois_ch = noiseprint(rgb.half()).float() * 255.
        dino_ch = dino_vit(rgb.half(), H, W).float() * 255.
    return torch.cat([imgs_10, srm_ch, nois_ch, dino_ch], dim=1)


def get_param_groups(model, base_lr, enc_ratio=0.1):
    raw      = model.module if hasattr(model, "module") else model
    enc_ids  = {id(p) for p in raw.encoder.parameters()}
    enc_p    = [p for p in raw.parameters() if id(p) in enc_ids]
    dec_p    = [p for p in raw.parameters() if id(p) not in enc_ids]
    return [
        {"params": enc_p, "lr": base_lr * enc_ratio},
        {"params": dec_p, "lr": base_lr},
    ]


def compute_metrics(preds_raw, masks, use_tta=False,
                    model=None, imgs_norm=None):
    if use_tta and imgs_norm is not None and model is not None:
        with torch.no_grad():
            p0 = torch.sigmoid(model(imgs_norm))
            p1 = torch.flip(torch.sigmoid(
                 model(torch.flip(imgs_norm,[-1]))),[-1])
            p2 = torch.flip(torch.sigmoid(
                 model(torch.flip(imgs_norm,[-2]))),[-2])
            p3 = torch.flip(torch.sigmoid(
                 model(torch.flip(imgs_norm,[-1,-2]))),[-1,-2])
        preds = ((p0+p1+p2+p3)/4. > 0.5)
    else:
        preds = (torch.sigmoid(preds_raw) > 0.5)
    masks = masks.bool()
    tp    = (preds  & masks ).float().sum((1,2,3))
    fp    = (preds  & ~masks).float().sum((1,2,3))
    fn    = (~preds & masks ).float().sum((1,2,3))
    iou   = (tp / (tp+fp+fn+1e-6)).mean().item()
    f1    = (2*tp / (2*tp+fp+fn+1e-6)).mean().item()
    prec  = (tp / (tp+fp+1e-6)).mean().item()
    rec   = (tp / (tp+fn+1e-6)).mean().item()
    return iou, prec, rec, f1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  — everything that spawns workers MUST be inside this block on Windows
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if CACHE_DIR is not None:
        CACHE_DIR.mkdir(exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    check_dataset(DATA_ROOT)

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"✓ GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB VRAM")

    print(f"\n{'='*64}")
    print(f"  Encoder       : {ENCODER}")
    print(f"  Channels      : {N_CH}")
    print(f"  Train samples : {TRAIN_SAMPLES:,}")
    print(f"  Val samples   : {VAL_SAMPLES:,}")
    print(f"  Batch         : {BATCH}  (×{GRAD_ACC} acc = {BATCH*GRAD_ACC} eff)")
    print(f"  Epochs        : {EPOCHS}  |  Patience : {PATIENCE}")
    print(f"  Image size    : {IMG_SIZE}")
    print(f"  Device        : {DEVICE}")
    print(f"{'='*64}\n")

    # ── Frozen GPU extractors ──────────────────────────────────────────────
    print("Loading frozen GPU extractors...")
    srm_net    = SRMExtractor().to(DEVICE).half().eval()
    noiseprint = NoiseprintExtractor().to(DEVICE).half().eval()
    dino_vit   = DinoViTExtractor().to(DEVICE).half().eval()
    print("✓ SRM / Noiseprint / DINO-ViT loaded\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    LMDB_TRAIN = DATA_ROOT / "DocTamperV1-TrainingSet"
    LMDB_VAL   = DATA_ROOT / "DocTamperV1-TestingSet"

    print("Running P-Hash deduplication...")
    train_idx = phash_deduplicate(LMDB_TRAIN, list(range(TRAIN_SAMPLES)))
    val_idx   = list(range(VAL_SAMPLES))

    train_ds = DocTamperDataset(LMDB_TRAIN, train_idx, _spatial_train,
                                 is_train=True, cache_dir=None)
    val_ds   = DocTamperDataset(LMDB_VAL,   val_idx,   _spatial_val,
                                 is_train=False, cache_dir=None)

    # Windows: num_workers > 0 requires if __name__ == '__main__'
    # We use 4 workers — safe on Windows with spawn
    NUM_WORKERS = 2

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          prefetch_factor=2, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True)

    print(f"✓ Train batches : {len(train_dl):,}")
    print(f"✓ Val   batches : {len(val_dl):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    try:
        model = smp.Unet(
            encoder_name    = ENCODER,
            encoder_weights = "imagenet",
            in_channels     = N_CH,
            classes         = 1,
            activation      = None,
        )
        print(f"✓ Encoder : {ENCODER}")
    except Exception as e:
        print(f"  {ENCODER} failed ({e}), using efficientnet-b5")
        model = smp.Unet(
            encoder_name    = "efficientnet-b5",
            encoder_weights = "imagenet",
            in_channels     = N_CH,
            classes         = 1,
            activation      = None,
        )
        print("✓ Encoder : efficientnet-b5 (fallback)")

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Params  : {n_params:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────
    dice_loss     = smp.losses.DiceLoss(mode="binary")
    bce_loss      = smp.losses.SoftBCEWithLogitsLoss(
                        pos_weight=torch.tensor([5.0]).to(DEVICE))
    focal_loss    = smp.losses.FocalLoss(mode="binary", gamma=2.5)
    tversky_loss  = TverskyLoss(alpha=0.3, beta=0.7).to(DEVICE)
    boundary_loss = BoundaryLoss().to(DEVICE)

    def loss_fn(preds, masks):
        return (0.30 * dice_loss(preds, masks)
              + 0.25 * tversky_loss(preds, masks)
              + 0.25 * focal_loss(preds, masks)
              + 0.10 * bce_loss(preds, masks)
              + 0.10 * boundary_loss(preds, masks))

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        get_param_groups(model, LR), weight_decay=WEIGHT_DECAY)
    steps = len(train_dl)
    warmup = LinearLR(optimizer, start_factor=0.05,
                      end_factor=1.0, total_iters=steps)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=(EPOCHS-1)*steps, eta_min=1e-7)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup, cosine],
                              milestones=[steps])
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    best_f1, epochs_no_improv = 0.0, 0
    history = []

    print("\n" + "="*66)
    print("  STARTING TRAINING")
    print(f"  Techniques: ELA · SRM · Noiseprint · DINO-ViT · OCR · P-Hash")
    print("="*66 + "\n")

    total_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (imgs_10, masks) in enumerate(train_dl):
            imgs_10 = imgs_10.to(DEVICE, non_blocking=True)
            masks   = masks.to(DEVICE, non_blocking=True)

            imgs_13 = gpu_features(imgs_10, DEVICE,
                                   srm_net, noiseprint, dino_vit)
            imgs_13 = normalise(imgs_13)

            with torch.amp.autocast("cuda"):
                preds = model(imgs_13)
                loss  = loss_fn(preds, masks) / GRAD_ACC

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * GRAD_ACC

        # Validate
        model.eval()
        val_loss = 0.0
        all_iou, all_p, all_r, all_f1 = [], [], [], []
        use_tta = (epoch > EPOCHS - 4)

        with torch.no_grad():
            for imgs_10, masks in val_dl:
                imgs_10 = imgs_10.to(DEVICE, non_blocking=True)
                masks   = masks.to(DEVICE, non_blocking=True)

                imgs_13 = gpu_features(imgs_10, DEVICE,
                                       srm_net, noiseprint, dino_vit)
                imgs_13 = normalise(imgs_13)

                with torch.amp.autocast("cuda"):
                    preds = model(imgs_13)
                    vloss = loss_fn(preds, masks)

                val_loss += vloss.item()
                iou, p, r, f1 = compute_metrics(
                    preds, masks,
                    use_tta  = use_tta,
                    model    = model,
                    imgs_norm= imgs_13 if use_tta else None,
                )
                all_iou.append(iou); all_p.append(p)
                all_r.append(r);     all_f1.append(f1)

        t_loss = train_loss / len(train_dl)
        v_loss = val_loss   / len(val_dl)
        m_iou  = float(np.mean(all_iou))
        m_f1   = float(np.mean(all_f1))
        m_p    = float(np.mean(all_p))
        m_r    = float(np.mean(all_r))
        lr_now = optimizer.param_groups[1]["lr"]
        ep_min = (time.time() - ep_t0) / 60
        tot_hr = (time.time() - total_t0) / 3600

        history.append(dict(epoch=epoch, train_loss=t_loss,
                            val_loss=v_loss, iou=m_iou,
                            f1=m_f1, precision=m_p, recall=m_r))

        star = " ★" if m_f1 > best_f1 else ""
        print(f"Ep {epoch:02d}/{EPOCHS} | "
              f"TLoss:{t_loss:.4f}  VLoss:{v_loss:.4f} | "
              f"IoU:{m_iou:.4f}  F1:{m_f1:.4f}  "
              f"P:{m_p:.4f}  R:{m_r:.4f} | "
              f"LR:{lr_now:.2e} | "
              f"{ep_min:.1f}min [{tot_hr:.2f}hr]{star}")

        if m_f1 > best_f1:
            best_f1 = m_f1
            epochs_no_improv = 0
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(),
                       OUTPUT_DIR / "forgery_best.pth")
            print(f"   ✓ forgery_best.pth saved  (F1={best_f1:.4f})")
        else:
            epochs_no_improv += 1
            print(f"   No gain ({epochs_no_improv}/{PATIENCE})")
            if epochs_no_improv >= PATIENCE:
                print(f"\n⚠  Early stop at epoch {epoch}")
                break

        if epoch % 5 == 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(),
                       OUTPUT_DIR / f"ckpt_ep{epoch:02d}_f1{m_f1:.3f}.pth")

        with open(OUTPUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    total_hrs = (time.time() - total_t0) / 3600
    print(f"\n{'='*66}")
    print(f"  DONE  |  Best F1 : {best_f1:.4f}  |  {total_hrs:.2f} hrs")
    print(f"  → ./output/forgery_best.pth")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
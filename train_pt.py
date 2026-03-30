from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from deepfillv2_masks_pt import sample_training_mask
from deepfillv2_model_pt import DeepFillV2Model
from deepfillv2_ops_pt import gan_hinge_loss

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for train_pt.py. Install it with: pip install pyyaml"
    ) from exc


RESIZE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def load_yaml_config(path: Path) -> Tuple[Dict[str, Any], Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict, dict_to_namespace(cfg_dict)


def resolve_flist(config: Any, dataset_name: str, which: str) -> Optional[str]:
    dataset_lists = getattr(config, "data_flist", None)
    if dataset_lists is None:
        return None
    if isinstance(dataset_lists, SimpleNamespace):
        dataset_lists = vars(dataset_lists)
    values = dataset_lists.get(dataset_name)
    if not values:
        return None
    if which == "train":
        return values[0]
    if which == "val" and len(values) > 1:
        return values[1]
    return None


def make_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_image_ram_gb(num_images: int, height: int, width: int, channels: int = 3) -> float:
    return float(num_images * height * width * channels) / float(1024 ** 3)


def read_flist(flist_path: Path) -> List[str]:
    flist_path = Path(flist_path)
    if not flist_path.is_file():
        raise FileNotFoundError(f"File list not found: {flist_path}")
    with flist_path.open("r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise RuntimeError(f"No image paths found in: {flist_path}")
    return paths


class ImageRAMDataset:
    """
    Preload the entire resized dataset into RAM as uint8 NCHW.

    This avoids disk reads and resize work inside the training loop.
    The tradeoff is memory: 200k images at 256x256 RGB uint8 are about 39 GiB raw.
    """

    def __init__(
        self,
        flist_path: Path,
        img_shapes: Iterable[int],
        *,
        random_crop: bool = False,
        resize_filter: str = "bilinear",
        progress_every: int = 1000,
    ) -> None:
        self.flist_path = Path(flist_path)
        self.paths = read_flist(self.flist_path)

        shapes = list(img_shapes)
        if len(shapes) < 2:
            raise ValueError(f"img_shapes must contain at least H,W, got: {img_shapes}")
        self.height = int(shapes[0])
        self.width = int(shapes[1])
        self.random_crop = bool(random_crop)
        if self.random_crop:
            raise NotImplementedError(
                "random_crop=True is not implemented because the exact neuralgym "
                "DataFromFNames cropping semantics were not provided. For the current "
                "celebahq-style config, use random_crop=False."
            )

        resize_filter = resize_filter.lower()
        if resize_filter not in RESIZE_FILTERS:
            raise ValueError(f"Unknown resize filter: {resize_filter}")
        self.resize_filter_name = resize_filter
        self.resize_filter = RESIZE_FILTERS[resize_filter]
        self.progress_every = max(1, int(progress_every))

        est_gb = estimate_image_ram_gb(len(self.paths), self.height, self.width, 3)
        print(
            f"Preloading {len(self.paths)} images from {self.flist_path} into RAM "
            f"as {self.height}x{self.width} uint8 RGB (~{est_gb:.2f} GiB raw)."
        )
        self.images_u8 = self._preload_images()

    def __len__(self) -> int:
        return self.images_u8.shape[0]

    def _load_image(self, path: str) -> Image.Image:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        with Image.open(p) as img:
            return img.convert("RGB")

    def _preload_images(self) -> torch.Tensor:
        n = len(self.paths)
        images = torch.empty((n, 3, self.height, self.width), dtype=torch.uint8)
        t0 = time.time()
        for i, path in enumerate(self.paths):
            img = self._load_image(path)
            img = img.resize((self.width, self.height), resample=self.resize_filter)
            arr = np.array(img, dtype=np.uint8, copy=True)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape} for {path}")
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            images[i].copy_(tensor)
            if (i + 1) % self.progress_every == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                ips = (i + 1) / max(elapsed, 1e-12)
                print(f"  loaded {i + 1}/{n} images ({ips:.1f} img/s)")
        return images


class RandomBatchIndexSource:
    def __init__(self, num_items: int, batch_size: int, *, seed: Optional[int] = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_items < batch_size:
            raise ValueError(
                f"Dataset of size {num_items} is smaller than batch_size {batch_size}; "
                "cannot emulate drop_last=True training."
            )
        self.num_items = int(num_items)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.order = torch.randperm(self.num_items, generator=self.generator)
        self.pos = 0

    def next(self) -> torch.Tensor:
        if self.pos + self.batch_size > self.num_items:
            self.order = torch.randperm(self.num_items, generator=self.generator)
            self.pos = 0
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx


class FixedVisualSet:
    def __init__(self, name: str, images_u8: torch.Tensor, masks: torch.Tensor) -> None:
        self.name = name
        self.images_u8 = images_u8.contiguous()
        self.masks = masks.contiguous()

    @property
    def count(self) -> int:
        return int(self.images_u8.shape[0])


class BatchStager:
    """
    CPU staging buffer for fast host->device copies.

    The dataset stays in ordinary RAM as uint8. For each step we gather the current batch
    into a fixed-size CPU buffer. If CUDA is used and pin_memory is enabled, that buffer is
    page-locked so the subsequent H2D copy can use non_blocking=True.

    We intentionally transfer uint8 to the GPU and convert to float there, reducing H2D
    traffic by 4x compared with copying float32 batches.
    """

    def __init__(self, batch_size: int, channels: int, height: int, width: int, *, pin_memory: bool) -> None:
        self.batch_size = int(batch_size)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.pin_memory = bool(pin_memory)
        self.buffer = torch.empty(
            (self.batch_size, self.channels, self.height, self.width),
            dtype=torch.uint8,
            pin_memory=self.pin_memory,
        )

    def stage_from(self, images_u8: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() != self.batch_size:
            raise ValueError(
                f"Expected exactly {self.batch_size} indices for the fixed staging buffer, got {indices.numel()}"
            )
        torch.index_select(images_u8, 0, indices, out=self.buffer)
        return self.buffer

    def to_device(self, device: torch.device) -> torch.Tensor:
        return self.buffer.to(device=device, non_blocking=(self.pin_memory and device.type == "cuda"))


class VisualStager:
    def __init__(self, images_u8: torch.Tensor, masks: torch.Tensor, *, pin_memory: bool) -> None:
        self.images_u8 = images_u8.pin_memory() if pin_memory else images_u8
        self.masks = masks.pin_memory() if pin_memory else masks
        self.pin_memory = pin_memory

    def images_to_device(self, device: torch.device) -> torch.Tensor:
        return self.images_u8.to(device=device, non_blocking=(self.pin_memory and device.type == "cuda"))

    def masks_to_device(self, device: torch.device) -> torch.Tensor:
        return self.masks.to(device=device, non_blocking=(self.pin_memory and device.type == "cuda"))


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1.0, 1.0)
    x = (x + 1.0) * 127.5
    x = x.permute(1, 2, 0).numpy()
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)



def make_mask_overlay(original_u8: np.ndarray, mask_1hw: torch.Tensor) -> np.ndarray:
    mask = (mask_1hw.detach().cpu().numpy()[0] > 0.5)
    overlay = original_u8.astype(np.float32).copy()
    tint = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * tint
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)



def save_triptych_grid(
    out_path: Path,
    originals_u8: List[np.ndarray],
    overlays_u8: List[np.ndarray],
    completes_u8: List[np.ndarray],
) -> None:
    rows: List[np.ndarray] = []
    for gt, ov, comp in zip(originals_u8, overlays_u8, completes_u8):
        rows.append(np.concatenate([gt, ov, comp], axis=1))
    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)



def batch_raw_to_batch_pos(batch_raw_u8: torch.Tensor) -> torch.Tensor:
    return batch_raw_u8.to(dtype=torch.float32) / 127.5 - 1.0



def run_fixed_visual_set(
    model: DeepFillV2Model,
    visual_set: FixedVisualSet,
    visual_stager: Optional[VisualStager],
    device: torch.device,
    out_path: Path,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if visual_stager is None:
            batch_raw_u8 = visual_set.images_u8.to(device=device)
            masks = visual_set.masks.to(device=device, dtype=torch.float32)
        else:
            batch_raw_u8 = visual_stager.images_to_device(device)
            masks = visual_stager.masks_to_device(device).to(dtype=torch.float32)

        batch_pos = batch_raw_to_batch_pos(batch_raw_u8)
        batch_incomplete = batch_pos * (1.0 - masks)
        _, x2, _ = model.generator(batch_incomplete, masks, return_flow=False)
        batch_complete = x2 * masks + batch_incomplete * (1.0 - masks)

        originals: List[np.ndarray] = []
        overlays: List[np.ndarray] = []
        completes: List[np.ndarray] = []
        for i in range(batch_raw_u8.shape[0]):
            gt = visual_set.images_u8[i].permute(1, 2, 0).cpu().numpy()
            comp = tensor_to_uint8_image(batch_complete[i])
            originals.append(gt)
            overlays.append(make_mask_overlay(gt, visual_set.masks[i]))
            completes.append(comp)

        save_triptych_grid(out_path, originals, overlays, completes)
    if was_training:
        model.train()



def sample_fixed_masks(config: Any, count: int, *, device: torch.device) -> torch.Tensor:
    masks = [
        sample_training_mask(config, device=device, dtype=torch.float32, batch_size=1)
        for _ in range(count)
    ]
    return torch.cat(masks, dim=0)



def build_fixed_visual_set(
    name: str,
    images_u8: torch.Tensor,
    config: Any,
    *,
    count: int,
) -> Optional[FixedVisualSet]:
    if count <= 0 or images_u8.shape[0] == 0:
        return None
    n = min(int(count), int(images_u8.shape[0]))
    chosen = images_u8[:n].clone()
    masks = sample_fixed_masks(config, n, device=torch.device("cpu"))
    return FixedVisualSet(name=name, images_u8=chosen, masks=masks)



def save_checkpoint(
    path: Path,
    *,
    model: DeepFillV2Model,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    step: int,
    resolved_config: Dict[str, Any],
    args_dict: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "resolved_config": resolved_config,
            "train_args": args_dict,
        },
        path,
    )



def save_latest_checkpoint(
    ckpt_dir: Path,
    *,
    model: DeepFillV2Model,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    step: int,
    resolved_config: Dict[str, Any],
    args_dict: Dict[str, Any],
) -> None:
    save_checkpoint(
        ckpt_dir / "latest.pt",
        model=model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        step=step,
        resolved_config=resolved_config,
        args_dict=args_dict,
    )



def train_step_d(
    model: DeepFillV2Model,
    batch_raw_u8: torch.Tensor,
    mask: torch.Tensor,
    d_optimizer: torch.optim.Optimizer,
) -> float:
    d_optimizer.zero_grad(set_to_none=True)

    batch_pos = batch_raw_to_batch_pos(batch_raw_u8)
    batch_incomplete = batch_pos * (1.0 - mask)
    _, x2, _ = model.generator(batch_incomplete, mask, return_flow=False)
    batch_complete = x2.detach() * mask + batch_incomplete * (1.0 - mask)

    disc_in = model.build_discriminator_input(
        batch_pos=batch_pos,
        batch_complete=batch_complete,
        masks=mask,
        edge=None,
    )
    pos_neg = model.discriminator(disc_in)
    pos, neg = torch.chunk(pos_neg, 2, dim=0)
    _, d_loss = gan_hinge_loss(pos, neg)
    d_loss.backward()
    d_optimizer.step()
    return float(d_loss.detach().cpu().item())



def train_step_g(
    model: DeepFillV2Model,
    batch_raw_u8: torch.Tensor,
    mask: torch.Tensor,
    g_optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    g_optimizer.zero_grad(set_to_none=True)

    batch_pos = batch_raw_to_batch_pos(batch_raw_u8)
    batch_incomplete = batch_pos * (1.0 - mask)
    x1, x2, _ = model.generator(batch_incomplete, mask, return_flow=False)
    batch_complete = x2 * mask + batch_incomplete * (1.0 - mask)

    ae_loss = model.l1_loss_alpha * torch.mean(torch.abs(batch_pos - x1))
    ae_loss = ae_loss + model.l1_loss_alpha * torch.mean(torch.abs(batch_pos - x2))

    disc_in = model.build_discriminator_input(
        batch_pos=batch_pos,
        batch_complete=batch_complete,
        masks=mask,
        edge=None,
    )
    pos_neg = model.discriminator(disc_in)
    pos, neg = torch.chunk(pos_neg, 2, dim=0)
    g_gan_loss, _ = gan_hinge_loss(pos, neg)

    g_loss = model.gan_loss_alpha * g_gan_loss
    if model.ae_loss:
        g_loss = g_loss + ae_loss

    g_loss.backward()
    g_optimizer.step()

    return {
        "g_loss": float(g_loss.detach().cpu().item()),
        "g_gan_loss": float(g_gan_loss.detach().cpu().item()),
        "ae_loss": float(ae_loss.detach().cpu().item()),
    }



def maybe_sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch training loop for the DeepFill v2 port.")
    parser.add_argument("--config", type=Path, default=Path("inpaint.yml"))
    parser.add_argument("--train_flist", type=Path, default=None)
    parser.add_argument("--val_flist", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset key from YAML")
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--static_train_count", type=int, default=6)
    parser.add_argument("--static_val_count", type=int, default=6)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resize_filter",
        type=str,
        default="bilinear",
        choices=sorted(RESIZE_FILTERS.keys()),
        help="Interpolation used for random_crop=False. The exact neuralgym interpolation was not visible in the provided files.",
    )
    parser.add_argument(
        "--preload_progress_every",
        type=int,
        default=1000,
        help="How often to print progress while resizing/loading images into RAM.",
    )
    parser.add_argument(
        "--pin_batch_staging",
        action="store_true",
        help="Use a pinned CPU staging buffer for batches before the host->device copy.",
    )
    parser.add_argument(
        "--pin_visual_staging",
        action="store_true",
        help="Pin the fixed train/val visualization sets for slightly faster host->device copies.",
    )
    args = parser.parse_args()

    config_dict, config = load_yaml_config(args.config)

    dataset_name = args.dataset or getattr(config, "dataset")
    train_flist = args.train_flist or resolve_flist(config, dataset_name, "train")
    val_flist = args.val_flist or resolve_flist(config, dataset_name, "val")
    if train_flist is None:
        raise ValueError("Could not resolve training .flist. Pass --train_flist explicitly.")

    if getattr(config, "guided", False):
        raise NotImplementedError(
            "guided=True training is not implemented in this script because the current CelebA setup is unguided and no edge pipeline was provided."
        )

    if getattr(config, "gan", "sngan") != "sngan":
        raise NotImplementedError("Only gan='sngan' is implemented, matching the provided config and model code.")

    batch_size = int(args.batch_size or getattr(config, "batch_size"))
    max_steps = int(args.max_steps or getattr(config, "max_iters"))
    save_every = int(args.save_every or getattr(config, "train_spe", 4000))
    sample_every = int(args.sample_every)
    print_every = int(args.print_every)
    device = make_device(args.device)

    log_dir = Path(args.log_dir or getattr(config, "log_dir", "logs/deepfillv2_pt"))
    ckpt_dir = log_dir / "checkpoints"
    sample_dir = log_dir / "samples"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    with (log_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    with (log_dir / "resolved_train_flist.txt").open("w", encoding="utf-8") as f:
        f.write(str(Path(train_flist)))
        f.write("\n")
    if val_flist is not None:
        with (log_dir / "resolved_val_flist.txt").open("w", encoding="utf-8") as f:
            f.write(str(Path(val_flist)))
            f.write("\n")

    train_data = ImageRAMDataset(
        Path(train_flist),
        getattr(config, "img_shapes"),
        random_crop=bool(getattr(config, "random_crop", False)),
        resize_filter=args.resize_filter,
        progress_every=args.preload_progress_every,
    )
    val_data = None
    if val_flist is not None and Path(val_flist).is_file():
        val_data = ImageRAMDataset(
            Path(val_flist),
            getattr(config, "img_shapes"),
            random_crop=bool(getattr(config, "random_crop", False)),
            resize_filter=args.resize_filter,
            progress_every=args.preload_progress_every,
        )

    height = int(getattr(config, "img_shapes")[0])
    width = int(getattr(config, "img_shapes")[1])
    pin_batch_staging = bool(args.pin_batch_staging and device.type == "cuda")
    pin_visual_staging = bool(args.pin_visual_staging and device.type == "cuda")

    batch_stager = BatchStager(batch_size, 3, height, width, pin_memory=pin_batch_staging)

    train_visuals = build_fixed_visual_set(
        "train",
        train_data.images_u8,
        config,
        count=args.static_train_count,
    )
    val_visuals = None
    if val_data is not None:
        val_visuals = build_fixed_visual_set(
            "val",
            val_data.images_u8,
            config,
            count=args.static_val_count,
        )

    train_visual_stager = None
    if train_visuals is not None and pin_visual_staging:
        train_visual_stager = VisualStager(train_visuals.images_u8, train_visuals.masks, pin_memory=True)
    val_visual_stager = None
    if val_visuals is not None and pin_visual_staging:
        val_visual_stager = VisualStager(val_visuals.images_u8, val_visuals.masks, pin_memory=True)

    model = DeepFillV2Model(
        guided=bool(getattr(config, "guided", False)),
        edge_threshold=float(getattr(config, "edge_threshold", 0.6)),
        padding=str(getattr(config, "padding", "SAME")),
        gan_with_mask=bool(getattr(config, "gan_with_mask", True)),
        gan_loss_alpha=float(getattr(config, "gan_loss_alpha", 1.0)),
        l1_loss_alpha=float(getattr(config, "l1_loss_alpha", 1.0)),
        ae_loss=bool(getattr(config, "ae_loss", True)),
    ).to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    lr = 1e-4
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        print(f"Resumed from {args.resume} at step {start_step}")

    save_latest_checkpoint(
        ckpt_dir,
        model=model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        step=start_step,
        resolved_config=config_dict,
        args_dict={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    if sample_every > 0:
        if train_visuals is not None:
            run_fixed_visual_set(
                model,
                train_visuals,
                train_visual_stager,
                device,
                sample_dir / f"train_fixed_step_{start_step:07d}.png",
            )
        if val_visuals is not None:
            run_fixed_visual_set(
                model,
                val_visuals,
                val_visual_stager,
                device,
                sample_dir / f"val_fixed_step_{start_step:07d}.png",
            )
        print(f"Saved initial fixed visualization grids at step {start_step}")

    batch_source = RandomBatchIndexSource(len(train_data), batch_size)
    steps_per_epoch = len(train_data) // batch_size
    print(
        f"Training set: {len(train_data)} images, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}."
    )

    if pin_batch_staging:
        print("Using pinned CPU staging buffer for training batches.")
    else:
        print("Using ordinary CPU staging buffer for training batches.")
    if pin_visual_staging:
        print("Using pinned CPU staging for fixed train/val visualizations.")

    model.train()
    maybe_sync_cuda(device)
    t0 = time.time()
    running: Dict[str, float] = {"g_loss": 0.0, "g_gan_loss": 0.0, "ae_loss": 0.0, "d_loss": 0.0}

    for step in range(start_step + 1, max_steps + 1):
        indices = batch_source.next()
        batch_stager.stage_from(train_data.images_u8, indices)
        batch_raw_u8 = batch_stager.to_device(device)
        mask = sample_training_mask(config, device=device, dtype=torch.float32, batch_size=batch_raw_u8.shape[0])

        d_loss = train_step_d(model, batch_raw_u8, mask, d_optimizer)
        g_stats = train_step_g(model, batch_raw_u8, mask, g_optimizer)

        running["d_loss"] += d_loss
        running["g_loss"] += g_stats["g_loss"]
        running["g_gan_loss"] += g_stats["g_gan_loss"]
        running["ae_loss"] += g_stats["ae_loss"]

        if step % print_every == 0:
            maybe_sync_cuda(device)
            elapsed = time.time() - t0
            avg = {k: v / print_every for k, v in running.items()}
            it_s = print_every / max(elapsed, 1e-12)
            current_epoch = (step - 1) // steps_per_epoch + 1
            epoch_step = ((step - 1) % steps_per_epoch) + 1
            epoch_progress = step / float(steps_per_epoch)
            print(
                f"step={step:07d} "
                f"epoch={current_epoch} "
                f"ep_step={epoch_step}/{steps_per_epoch} "
                f"epoch_prog={epoch_progress:.3f} "
                f"g={avg['g_loss']:.4f} "
                f"adv={avg['g_gan_loss']:.4f} "
                f"l1={avg['ae_loss']:.4f} "
                f"d={avg['d_loss']:.4f} "
                f"iter/s={it_s:.3f}"
            )
            for k in running:
                running[k] = 0.0
            t0 = time.time()

        if sample_every > 0 and step % sample_every == 0:
            if train_visuals is not None:
                run_fixed_visual_set(
                    model,
                    train_visuals,
                    train_visual_stager,
                    device,
                    sample_dir / f"train_fixed_step_{step:07d}.png",
                )
            if val_visuals is not None:
                run_fixed_visual_set(
                    model,
                    val_visuals,
                    val_visual_stager,
                    device,
                    sample_dir / f"val_fixed_step_{step:07d}.png",
                )
            print(f"Saved fixed visualization grids at step {step}")

        if save_every > 0 and step % save_every == 0:
            step_path = ckpt_dir / f"step_{step:07d}.pt"
            args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            save_checkpoint(
                step_path,
                model=model,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                step=step,
                resolved_config=config_dict,
                args_dict=args_dict,
            )
            save_latest_checkpoint(
                ckpt_dir,
                model=model,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                step=step,
                resolved_config=config_dict,
                args_dict=args_dict,
            )
            print(f"Saved checkpoint at step {step}: {step_path}")


if __name__ == "__main__":
    main()

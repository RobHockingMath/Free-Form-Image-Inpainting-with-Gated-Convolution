from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw


# Faithful to the original TensorFlow file, which seeds numpy at import time.
np.random.seed(2018)


@dataclass
class MaskConfig:
    """
    Minimal configuration needed by the original mask-generation code.

    This mirrors the FLAGS fields consumed by:
      - random_bbox
      - bbox2mask
      - brush_stroke_mask

    img_shapes is expected to be [H, W, C].
    """

    img_shapes: Tuple[int, int, int]
    height: int = 128
    width: int = 128
    max_delta_height: int = 32
    max_delta_width: int = 32
    vertical_margin: int = 0
    horizontal_margin: int = 0


FlagsLike = Union[MaskConfig, Dict[str, Any], Any]
BBox = Tuple[int, int, int, int]


def _flag(flags: FlagsLike, name: str) -> Any:
    if isinstance(flags, dict):
        return flags[name]
    return getattr(flags, name)


def _img_hw(flags: FlagsLike) -> Tuple[int, int]:
    img_shapes = _flag(flags, "img_shapes")
    if len(img_shapes) < 2:
        raise ValueError(f"img_shapes must have at least two entries, got {img_shapes}")
    return int(img_shapes[0]), int(img_shapes[1])


def random_bbox(flags: FlagsLike) -> BBox:
    """
    Faithful port of inpaint_ops.random_bbox.

    Original TensorFlow:
        maxt = img_height - FLAGS.vertical_margin - FLAGS.height
        maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
        t = tf.random_uniform([], minval=FLAGS.vertical_margin, maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform([], minval=FLAGS.horizontal_margin, maxval=maxl, dtype=tf.int32)

    Important:
      - maxval is exclusive in TensorFlow for integer uniform sampling.
      - this returns a single bbox, not one bbox per image in a batch.
    """
    img_height, img_width = _img_hw(flags)
    box_h = int(_flag(flags, "height"))
    box_w = int(_flag(flags, "width"))
    vmargin = int(_flag(flags, "vertical_margin"))
    hmargin = int(_flag(flags, "horizontal_margin"))

    maxt = img_height - vmargin - box_h
    maxl = img_width - hmargin - box_w

    if maxt <= vmargin:
        raise ValueError(
            f"Invalid bbox vertical range: min={vmargin}, max(exclusive)={maxt}. "
            f"img_height={img_height}, box_h={box_h}, vertical_margin={vmargin}"
        )
    if maxl <= hmargin:
        raise ValueError(
            f"Invalid bbox horizontal range: min={hmargin}, max(exclusive)={maxl}. "
            f"img_width={img_width}, box_w={box_w}, horizontal_margin={hmargin}"
        )

    top = int(np.random.randint(vmargin, maxt))
    left = int(np.random.randint(hmargin, maxl))
    return (top, left, box_h, box_w)


def _bbox2mask_numpy(
    bbox: BBox,
    height: int,
    width: int,
    delta_h: int,
    delta_w: int,
) -> np.ndarray:
    """
    Faithful numpy port of the nested npmask(...) in inpaint_ops.bbox2mask.

    Returns NHWC shape [1, H, W, 1] to mirror the original helper internally.
    """
    mask = np.zeros((1, height, width, 1), np.float32)
    h = int(np.random.randint(delta_h // 2 + 1))
    w = int(np.random.randint(delta_w // 2 + 1))
    mask[:, bbox[0] + h:bbox[0] + bbox[2] - h, bbox[1] + w:bbox[1] + bbox[3] - w, :] = 1.0
    return mask


def bbox2mask(
    flags: FlagsLike,
    bbox: BBox,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Port of inpaint_ops.bbox2mask.

    Returns NCHW [1, 1, H, W] for direct use in the PyTorch model.
    The original TensorFlow function returned NHWC [1, H, W, 1].
    """
    height, width = _img_hw(flags)
    mask = _bbox2mask_numpy(
        bbox=bbox,
        height=height,
        width=width,
        delta_h=int(_flag(flags, "max_delta_height")),
        delta_w=int(_flag(flags, "max_delta_width")),
    )
    mask_t = torch.from_numpy(mask).permute(0, 3, 1, 2).contiguous()
    if device is not None:
        mask_t = mask_t.to(device=device)
    return mask_t.to(dtype=dtype)


def _brush_stroke_mask_numpy(height: int, width: int) -> np.ndarray:
    """
    Faithful numpy/PIL port of the nested generate_mask(H, W) in inpaint_ops.brush_stroke_mask.

    Important fidelity notes:
      1. The original code does *not* assign the result of mask.transpose(...),
         so the flip calls are effectively no-ops. This port preserves that.
      2. The original code does:
             h, w = mask.size
         even though PIL returns (width, height). This only matters for non-square
         images, but we preserve it exactly.

    Returns NHWC shape [1, H, W, 1] internally, matching the TF helper.
    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 12
    max_width = 40

    average_radius = math.sqrt(height * height + width * width) / 8
    mask = Image.new("L", (width, height), 0)

    for _ in range(int(np.random.randint(1, 4))):
        num_vertex = int(np.random.randint(min_num_vertex, max_num_vertex))
        angle_min = mean_angle - float(np.random.uniform(0, angle_range))
        angle_max = mean_angle + float(np.random.uniform(0, angle_range))
        angles = []
        vertex = []

        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - float(np.random.uniform(angle_min, angle_max)))
            else:
                angles.append(float(np.random.uniform(angle_min, angle_max)))

        # Preserve the original code exactly, including the swapped variable names.
        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0,
                2 * average_radius,
            )
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        brush_width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=brush_width)
        for v in vertex:
            draw.ellipse(
                (
                    v[0] - brush_width // 2,
                    v[1] - brush_width // 2,
                    v[0] + brush_width // 2,
                    v[1] + brush_width // 2,
                ),
                fill=1,
            )

    # Preserve the original no-op bug exactly: transpose returns a new image,
    # but the original code discards it.
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, height, width, 1))
    return mask


def brush_stroke_mask(
    flags: FlagsLike,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Port of inpaint_ops.brush_stroke_mask.

    Returns NCHW [1, 1, H, W] for direct use in the PyTorch model.
    The original TensorFlow function returned NHWC [1, H, W, 1].
    """
    height, width = _img_hw(flags)
    mask = _brush_stroke_mask_numpy(height, width)
    mask_t = torch.from_numpy(mask).permute(0, 3, 1, 2).contiguous()
    if device is not None:
        mask_t = mask_t.to(device=device)
    return mask_t.to(dtype=dtype)


def sample_training_mask(
    flags: FlagsLike,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Faithful port of the training-time mask logic from
    InpaintCAModel.build_graph_with_losses:

        bbox = random_bbox(FLAGS)
        regular_mask = bbox2mask(FLAGS, bbox)
        irregular_mask = brush_stroke_mask(FLAGS)
        mask = logical_or(irregular_mask, regular_mask)

    Returns:
        mask: [1, 1, H, W] by default, or [B, 1, H, W] if batch_size is given.

    Important:
      - The original TensorFlow graph creates a single [1, H, W, 1] mask and
        relies on broadcasting across the batch. Passing batch_size here repeats
        that same single sampled mask across the batch, matching the original behavior.
    """
    bbox = random_bbox(flags)
    regular_mask = bbox2mask(flags, bbox, device=device, dtype=dtype)
    irregular_mask = brush_stroke_mask(flags, device=device, dtype=dtype)
    mask = torch.logical_or(irregular_mask.bool(), regular_mask.bool()).to(dtype=dtype)

    if batch_size is not None:
        mask = mask.repeat(int(batch_size), 1, 1, 1)

    return mask


__all__ = [
    "MaskConfig",
    "random_bbox",
    "bbox2mask",
    "brush_stroke_mask",
    "sample_training_mask",
]

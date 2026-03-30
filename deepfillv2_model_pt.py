# deepfillv2_model_pt.py
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfillv2_ops_pt import (
    GenConv2d,
    GenDeconv2d,
    SNPatchDiscriminator,
    gan_hinge_loss,
    resize_nearest,
    resize_bilinear,
    resize_mask_like,
    resize_nearest_tf_align_corners,
    tf_same_pad_2d,
)

IntOrPair = Union[int, Tuple[int, int]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def gated_out_channels(cnum: int, activation_is_none: bool = False) -> int:
    """
    Mirrors inpaint_ops.gen_conv:
      - if cnum == 3 or activation is None: output has cnum channels
      - otherwise output has cnum // 2 channels after feature/gate split
    """
    if cnum == 3 or activation_is_none:
        return cnum
    if cnum % 2 != 0:
        raise ValueError(f"gated conv requires even raw channel count, got {cnum}")
    return cnum // 2


def extract_image_patches_same(
    x: torch.Tensor,
    kernel_size: IntOrPair,
    stride: IntOrPair = 1,
) -> torch.Tensor:
    """
    NCHW equivalent of tf.extract_image_patches(..., padding='SAME')
    for the cases used by contextual_attention.

    Returns:
        patches: [B, L, C, KH, KW]
    """
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)

    x = tf_same_pad_2d(x, (kh, kw), (sh, sw), (1, 1))
    patches = F.unfold(x, kernel_size=(kh, kw), dilation=1, padding=0, stride=(sh, sw))
    b, ck2, l = patches.shape
    c = x.shape[1]
    patches = patches.transpose(1, 2).contiguous().view(b, l, c, kh, kw)
    return patches


def conv2d_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: IntOrPair = 1,
    dilation: IntOrPair = 1,
) -> torch.Tensor:
    kh, kw = weight.shape[-2:]
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    x = tf_same_pad_2d(x, (kh, kw), (sh, sw), (dh, dw))
    return F.conv2d(x, weight, bias=None, stride=(sh, sw), padding=0, dilation=(dh, dw))


def conv_transpose2d_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: IntOrPair,
) -> torch.Tensor:
    """
    PyTorch conv_transpose2d wrapper chosen to reproduce TF SAME behavior
    for the exact case used in contextual attention.

    TF call:
        tf.nn.conv2d_transpose(yi, wi_center, output_shape, strides=[1, rate, rate, 1], padding='SAME')

    For kernel = 2*rate and stride = rate, this padding/output_padding formula
    reproduces the requested size.
    """
    sh, sw = _pair(stride)
    kh, kw = weight.shape[-2:]

    pad_h = math.ceil((kh - sh) / 2)
    pad_w = math.ceil((kw - sw) / 2)
    out_pad_h = 2 * pad_h - (kh - sh)
    out_pad_w = 2 * pad_w - (kw - sw)

    if not (0 <= out_pad_h < sh and 0 <= out_pad_w < sw):
        raise ValueError(
            f"Invalid output padding for transposed SAME conv: "
            f"kernel=({kh},{kw}) stride=({sh},{sw}) -> output_padding=({out_pad_h},{out_pad_w})"
        )

    return F.conv_transpose2d(
        x,
        weight,
        bias=None,
        stride=(sh, sw),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
    )


def _match_spatial_size(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Safety clamp to exact target size if transposed-conv rounding differs by 1.
    The common 256x256 / rate=2 path should already match exactly.
    """
    th, tw = target_hw
    h, w = x.shape[-2:]

    if h > th:
        x = x[:, :, :th, :]
    elif h < th:
        x = F.pad(x, (0, 0, 0, th - h))

    if w > tw:
        x = x[:, :, :, :tw]
    elif w < tw:
        x = F.pad(x, (0, tw - w, 0, 0))

    return x


def make_color_wheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = (15, 6, 4, 11, 13, 6)
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry

    colorwheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col:col + yg, 1] = 255
    col += yg

    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc

    colorwheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(0, cb) / cb)
    colorwheel[col:col + cb, 2] = 255
    col += cb

    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm

    colorwheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(0, mr) / mr)
    colorwheel[col:col + mr, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = u.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    nan_idx = np.isnan(u) | np.isnan(v)
    u = u.copy()
    v = v.copy()
    u[nan_idx] = 0
    v[nan_idx] = 0

    colorwheel = COLORWHEEL
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255.0
        col1 = tmp[k1 - 1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])

        notidx = ~idx
        col[notidx] *= 0.75

        img[:, :, i] = np.floor(255 * col * (~nan_idx)).astype(np.uint8)

    return img


def flow_to_image(flow: np.ndarray) -> np.ndarray:
    """
    Port of inpaint_ops.flow_to_image.
    Expects flow as [B, H, W, 2].
    Returns uint8-ish float array [B, H, W, 3].
    """
    out = []
    maxrad = -1.0

    # First pass to find a global maxrad, matching the TF helper's per-batch loop behavior.
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0].copy()
        v = flow[i, :, :, 1].copy()
        idx_unknown = (np.abs(u) > 1e7) | (np.abs(v) > 1e7)
        u[idx_unknown] = 0
        v[idx_unknown] = 0
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))

    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0].copy()
        v = flow[i, :, :, 1].copy()
        idx_unknown = (np.abs(u) > 1e7) | (np.abs(v) > 1e7)
        u[idx_unknown] = 0
        v[idx_unknown] = 0

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)

    return np.float32(np.uint8(out))


def flow_to_image_torch(flow_bhw2: torch.Tensor) -> torch.Tensor:
    """
    Converts [B, H, W, 2] offsets to an NCHW image in [-1, 1],
    matching the TF helper path used only for summaries/visualization.
    """
    flow_np = flow_bhw2.detach().cpu().numpy()
    img_np = flow_to_image(flow_np)  # [B, H, W, 3]
    img = torch.from_numpy(img_np).to(device=flow_bhw2.device, dtype=torch.float32)
    img = img.permute(0, 3, 1, 2).contiguous()
    img = img / 127.5 - 1.0
    return img


class ContextualAttention(nn.Module):
    """
    PyTorch port of inpaint_ops.contextual_attention.

    Notes:
      - Uses NCHW instead of NHWC.
      - Keeps the same two-level patch extraction:
          raw_w on original-resolution background with kernel = 2*rate
          w     on downscaled background with kernel = ksize
      - Keeps the same score-fusing logic.
      - Keeps the hardcoded "/ 4." after conv_transpose2d.
    """

    def __init__(
        self,
        ksize: int = 3,
        stride: int = 1,
        rate: int = 2,
        fuse_k: int = 3,
        softmax_scale: float = 10.0,
        fuse: bool = True,
    ) -> None:
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse

    def forward(
        self,
        f: torch.Tensor,
        b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_flow: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw_h, raw_w = f.shape[-2:]
        b_raw_h, b_raw_w = b.shape[-2:]
        batch_size = f.shape[0]
        channels = f.shape[1]

        kernel = 2 * self.rate

        # raw_w: patches from original-resolution background, stride = rate * stride
        raw_w_patches = extract_image_patches_same(
            b,
            kernel_size=kernel,
            stride=self.rate * self.stride,
        )  # [B, L, C, kernel, kernel]

        # downscale foreground/background for matching
        f_down = resize_nearest_tf_align_corners(
            f, (raw_h // self.rate, raw_w // self.rate)
        )
        b_down = resize_nearest_tf_align_corners(
            b, (b_raw_h // self.rate, b_raw_w // self.rate)
        )
        if mask is not None:
            mask_down = resize_nearest_tf_align_corners(
                mask,
                (mask.shape[-2] // self.rate, mask.shape[-1] // self.rate),
            )
        else:
            mask_down = None

        fs_h, fs_w = f_down.shape[-2:]
        bs_h, bs_w = b_down.shape[-2:]

        # matching patches on downscaled background
        w_patches = extract_image_patches_same(
            b_down,
            kernel_size=self.ksize,
            stride=self.stride,
        )  # [B, L, C, k, k]

        if mask_down is None:
            mask_down = torch.zeros(
                (1, 1, bs_h, bs_w),
                dtype=b_down.dtype,
                device=b_down.device,
            )

        m_patches = extract_image_patches_same(
            mask_down,
            kernel_size=self.ksize,
            stride=self.stride,
        )  # [Bm, L, 1, k, k]

        fuse_weight = torch.eye(
            self.fuse_k,
            dtype=f_down.dtype,
            device=f_down.device,
        ).view(1, 1, self.fuse_k, self.fuse_k)

        ys = []
        offsets = []

        for i in range(batch_size):
            xi = f_down[i: i + 1]          # [1, C, fs_h, fs_w]
            wi = w_patches[i]              # [L, C, k, k]
            raw_wi = raw_w_patches[i]      # [L, C, kernel, kernel]

            # mask: use first if only one mask patch set exists, otherwise per-sample
            mi = m_patches[0 if m_patches.shape[0] == 1 else i]  # [L, 1, k, k]
            mm = (mi.mean(dim=(1, 2, 3), keepdim=True) == 0.0).to(xi.dtype)  # [L,1,1,1]
            mm = mm.view(1, -1, 1, 1)  # [1, L, 1, 1]

            # normalize each background patch
            wi_norm = wi / torch.clamp(
                torch.sqrt(torch.sum(wi ** 2, dim=(1, 2, 3), keepdim=True)),
                min=1e-4,
            )

            # match: conv foreground against normalized background patches
            yi = conv2d_same(xi, wi_norm, stride=1, dilation=1)  # [1, L, fs_h, fs_w]

            # fuse scores to encourage large patches
            if self.fuse:
                yi_nhwc = yi.permute(0, 2, 3, 1).contiguous()  # [1, fs_h, fs_w, L]

                temp = yi_nhwc.view(1, fs_h * fs_w, bs_h * bs_w, 1).permute(0, 3, 1, 2)
                temp = conv2d_same(temp, fuse_weight, stride=1, dilation=1)
                temp = temp.permute(0, 2, 3, 1).contiguous().view(1, fs_h, fs_w, bs_h, bs_w)

                temp = temp.permute(0, 2, 1, 4, 3).contiguous()
                temp = temp.view(1, fs_h * fs_w, bs_h * bs_w, 1).permute(0, 3, 1, 2)
                temp = conv2d_same(temp, fuse_weight, stride=1, dilation=1)
                temp = temp.permute(0, 2, 3, 1).contiguous().view(1, fs_w, fs_h, bs_w, bs_h)

                temp = temp.permute(0, 2, 1, 4, 3).contiguous()
                yi = temp.view(1, fs_h, fs_w, bs_h * bs_w).permute(0, 3, 1, 2).contiguous()

            # mask and softmax
            yi = yi * mm
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mm

            # argmax offsets in the downsampled background grid
            offset = torch.argmax(yi, dim=1)  # [1, fs_h, fs_w]
            offset = torch.stack(
                [offset // fs_w, offset % fs_w],
                dim=-1,
            )  # [1, fs_h, fs_w, 2]

            # paste center patches back to original resolution
            yi = conv_transpose2d_same(yi, raw_wi, stride=self.rate) / 4.0
            yi = _match_spatial_size(yi, (raw_h, raw_w))

            ys.append(yi)
            offsets.append(offset)

        y = torch.cat(ys, dim=0)
        offsets = torch.cat(offsets, dim=0)  # [B, fs_h, fs_w, 2]

        h_add = torch.arange(bs_h, device=offsets.device, dtype=offsets.dtype).view(1, bs_h, 1, 1)
        h_add = h_add.expand(offsets.shape[0], bs_h, bs_w, 1)
        w_add = torch.arange(bs_w, device=offsets.device, dtype=offsets.dtype).view(1, 1, bs_w, 1)
        w_add = w_add.expand(offsets.shape[0], bs_h, bs_w, 1)
        offsets = offsets - torch.cat([h_add, w_add], dim=3)

        flow = None
        if return_flow:
            flow = flow_to_image_torch(offsets)
            if self.rate != 1:
                flow = resize_bilinear(flow, scale=float(self.rate))

        return y, flow


class InpaintGenerator(nn.Module):
    """
    PyTorch port of InpaintCAModel.build_inpaint_net.

    Input:
        xin  : incomplete image in [-1, 1], shape [B, 3, H, W] if unguided
               or [B, 4, H, W] if guided (RGB + edge)
        mask : {0,1}, shape [B, 1, H, W], where 1 means masked region

    Returns:
        x_stage1, x_stage2, offset_flow
    """

    def __init__(self, guided: bool = False, padding: str = "SAME") -> None:
        super().__init__()
        self.guided = guided
        self.padding = padding

        base_in = 4 if guided else 3
        cnum = 48

        c1 = gated_out_channels(cnum)
        c2 = gated_out_channels(2 * cnum)
        c4 = gated_out_channels(4 * cnum)
        c16 = gated_out_channels(cnum // 2)  # conv16 raw = 24 -> out = 12

        # Stage 1
        self.conv1 = GenConv2d(base_in + 2, cnum, 5, 1, padding=padding)
        self.conv2_downsample = GenConv2d(c1, 2 * cnum, 3, 2, padding=padding)
        self.conv3 = GenConv2d(c2, 2 * cnum, 3, 1, padding=padding)
        self.conv4_downsample = GenConv2d(c2, 4 * cnum, 3, 2, padding=padding)
        self.conv5 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.conv6 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.conv7_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=2, padding=padding)
        self.conv8_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=4, padding=padding)
        self.conv9_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=8, padding=padding)
        self.conv10_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=16, padding=padding)
        self.conv11 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.conv12 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.conv13_upsample = GenDeconv2d(c4, 2 * cnum, padding=padding)
        self.conv14 = GenConv2d(c2, 2 * cnum, 3, 1, padding=padding)
        self.conv15_upsample = GenDeconv2d(c2, cnum, padding=padding)
        self.conv16 = GenConv2d(c1, cnum // 2, 3, 1, padding=padding)
        self.conv17 = GenConv2d(c16, 3, 3, 1, padding=padding, activation=None)

        # Stage 2 - conv branch
        self.xconv1 = GenConv2d(3, cnum, 5, 1, padding=padding)
        self.xconv2_downsample = GenConv2d(c1, cnum, 3, 2, padding=padding)
        self.xconv3 = GenConv2d(c1, 2 * cnum, 3, 1, padding=padding)
        self.xconv4_downsample = GenConv2d(c2, 2 * cnum, 3, 2, padding=padding)
        self.xconv5 = GenConv2d(c2, 4 * cnum, 3, 1, padding=padding)
        self.xconv6 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.xconv7_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=2, padding=padding)
        self.xconv8_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=4, padding=padding)
        self.xconv9_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=8, padding=padding)
        self.xconv10_atrous = GenConv2d(c4, 4 * cnum, 3, 1, rate=16, padding=padding)

        # Stage 2 - attention branch
        self.pmconv1 = GenConv2d(3, cnum, 5, 1, padding=padding)
        self.pmconv2_downsample = GenConv2d(c1, cnum, 3, 2, padding=padding)
        self.pmconv3 = GenConv2d(c1, 2 * cnum, 3, 1, padding=padding)
        self.pmconv4_downsample = GenConv2d(c2, 4 * cnum, 3, 2, padding=padding)
        self.pmconv5 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.pmconv6 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding, activation=F.relu)
        self.contextual_attention = ContextualAttention(
            ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10.0, fuse=True
        )
        self.pmconv9 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.pmconv10 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)

        # Stage 2 - merge branch
        self.allconv11 = GenConv2d(c4 + c4, 4 * cnum, 3, 1, padding=padding)
        self.allconv12 = GenConv2d(c4, 4 * cnum, 3, 1, padding=padding)
        self.allconv13_upsample = GenDeconv2d(c4, 2 * cnum, padding=padding)
        self.allconv14 = GenConv2d(c2, 2 * cnum, 3, 1, padding=padding)
        self.allconv15_upsample = GenDeconv2d(c2, cnum, padding=padding)
        self.allconv16 = GenConv2d(c1, cnum // 2, 3, 1, padding=padding)
        self.allconv17 = GenConv2d(c16, 3, 3, 1, padding=padding, activation=None)

    def forward(
        self,
        xin: torch.Tensor,
        mask: torch.Tensor,
        return_flow: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        xin_orig = xin
        ones_x = torch.ones_like(xin[:, :1])
        x = torch.cat([xin, ones_x, ones_x * mask], dim=1)

        # Stage 1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask_s = resize_mask_like(mask, x)

        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = torch.tanh(x)

        # Stage 2: paste stage1 result into RGB only
        x = x_stage1 * mask + xin_orig[:, :3] * (1.0 - mask)

        # Conv branch
        xnow = x
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        # Attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextual_attention(x, x, mask_s, return_flow=return_flow)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x

        # Merge
        x = torch.cat([x_hallu, pm], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)

        return x_stage1, x_stage2, offset_flow


class DeepFillV2Model(nn.Module):
    """
    Thin PyTorch wrapper around the generator + SN-PatchGAN discriminator,
    mirroring the visible TensorFlow model behavior.

    This does not reproduce neuralgym training infrastructure.
    It does reproduce:
      - preprocessing from raw image / mask / edge tensors
      - batch_complete assembly
      - discriminator input assembly
      - stage1/stage2 L1 loss
      - hinge GAN loss
    """

    def __init__(
        self,
        guided: bool = False,
        edge_threshold: float = 0.6,
        padding: str = "SAME",
        gan_with_mask: bool = True,
        gan_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 1.0,
        ae_loss: bool = True,
    ) -> None:
        super().__init__()
        self.guided = guided
        self.edge_threshold = edge_threshold
        self.gan_with_mask = gan_with_mask
        self.gan_loss_alpha = gan_loss_alpha
        self.l1_loss_alpha = l1_loss_alpha
        self.ae_loss = ae_loss

        self.generator = InpaintGenerator(guided=guided, padding=padding)

        disc_in_channels = 3
        if gan_with_mask:
            disc_in_channels += 1
        if guided:
            disc_in_channels += 1
        self.discriminator = SNPatchDiscriminator(disc_in_channels)

    @staticmethod
    def _maybe_broadcast_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if mask.shape[0] == batch_size:
            return mask
        if mask.shape[0] == 1:
            return mask.repeat(batch_size, 1, 1, 1)
        raise ValueError(f"Mask batch dim {mask.shape[0]} does not match batch size {batch_size}")

    @staticmethod
    def _repeat_for_pos_neg(x: torch.Tensor, total_batch: int) -> torch.Tensor:
        if x.shape[0] == total_batch:
            return x
        if x.shape[0] * 2 == total_batch:
            return x.repeat(2, 1, 1, 1)
        if x.shape[0] == 1:
            return x.repeat(total_batch, 1, 1, 1)
        raise ValueError(f"Cannot tile tensor of batch {x.shape[0]} to total batch {total_batch}")

    def prepare_inputs_from_raw(
        self,
        batch_raw: torch.Tensor,
        mask_raw: torch.Tensor,
        edge_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Mirrors build_server_graph preprocessing, but accepts separate tensors.

        Inputs:
            batch_raw : [B, 3, H, W], raw 0..255
            mask_raw  : [B or 1, 1 or 3, H, W], raw 0..255, >127.5 means masked
            edge_raw  : [B or 1, 1 or 3, H, W], raw 0..255, optional

        Returns:
            batch_pos       : normalized full image in [-1,1]
            masks           : {0,1} [B,1,H,W]
            edge            : thresholded {0,1} * mask or None
            batch_incomplete: normalized incomplete RGB in [-1,1]
        """
        batch_size = batch_raw.shape[0]

        masks = (mask_raw[:, :1] > 127.5).to(batch_raw.dtype)
        masks = self._maybe_broadcast_mask(masks, batch_size)

        batch_pos = batch_raw / 127.5 - 1.0
        batch_incomplete = batch_pos * (1.0 - masks)

        edge = None
        if self.guided:
            if edge_raw is None:
                raise ValueError("guided=True requires edge_raw")
            edge = edge_raw[:, :1] / 255.0
            edge = (edge > self.edge_threshold).to(batch_raw.dtype)
            edge = self._maybe_broadcast_mask(edge, batch_size)
            edge = edge * masks

        return batch_pos, masks, edge, batch_incomplete

    def inpaint_from_raw(
        self,
        batch_raw: torch.Tensor,
        mask_raw: torch.Tensor,
        edge_raw: Optional[torch.Tensor] = None,
        return_flow: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_pos, masks, edge, batch_incomplete = self.prepare_inputs_from_raw(
            batch_raw=batch_raw,
            mask_raw=mask_raw,
            edge_raw=edge_raw,
        )

        if self.guided:
            xin = torch.cat([batch_incomplete, edge], dim=1)
        else:
            xin = batch_incomplete

        x1, x2, flow = self.generator(xin, masks, return_flow=return_flow)
        batch_complete = x2 * masks + batch_incomplete * (1.0 - masks)

        return {
            "batch_pos": batch_pos,
            "masks": masks,
            "edge": edge,
            "batch_incomplete": batch_incomplete,
            "x_stage1": x1,
            "x_stage2": x2,
            "batch_complete": batch_complete,
            "flow": flow,
        }

    def build_discriminator_input(
        self,
        batch_pos: torch.Tensor,
        batch_complete: torch.Tensor,
        masks: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mirrors build_graph_with_losses discriminator assembly.
        """
        batch_pos_neg = torch.cat([batch_pos, batch_complete], dim=0)

        if self.gan_with_mask:
            mask_for_disc = self._repeat_for_pos_neg(masks, batch_pos_neg.shape[0])
            batch_pos_neg = torch.cat([batch_pos_neg, mask_for_disc], dim=1)

        if self.guided:
            if edge is None:
                raise ValueError("guided discriminator input requires edge")
            edge_for_disc = self._repeat_for_pos_neg(edge, batch_pos_neg.shape[0])
            batch_pos_neg = torch.cat([batch_pos_neg, edge_for_disc], dim=1)

        return batch_pos_neg

    def forward_train(
        self,
        batch_raw: torch.Tensor,
        mask_raw: torch.Tensor,
        edge_raw: Optional[torch.Tensor] = None,
        return_flow: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        End-to-end forward pass with losses, mirroring build_graph_with_losses.
        """
        out = self.inpaint_from_raw(
            batch_raw=batch_raw,
            mask_raw=mask_raw,
            edge_raw=edge_raw,
            return_flow=return_flow,
        )

        batch_pos = out["batch_pos"]
        masks = out["masks"]
        edge = out["edge"]
        batch_complete = out["batch_complete"]
        x1 = out["x_stage1"]
        x2 = out["x_stage2"]

        ae_loss = self.l1_loss_alpha * torch.mean(torch.abs(batch_pos - x1))
        ae_loss = ae_loss + self.l1_loss_alpha * torch.mean(torch.abs(batch_pos - x2))

        disc_in = self.build_discriminator_input(
            batch_pos=batch_pos,
            batch_complete=batch_complete,
            masks=masks,
            edge=edge,
        )
        pos_neg = self.discriminator(disc_in)
        pos, neg = torch.chunk(pos_neg, 2, dim=0)

        g_gan_loss, d_loss = gan_hinge_loss(pos, neg)
        g_loss = self.gan_loss_alpha * g_gan_loss
        if self.ae_loss:
            g_loss = g_loss + ae_loss

        out.update(
            {
                "ae_loss": ae_loss,
                "g_gan_loss": g_gan_loss,
                "g_loss": g_loss,
                "d_loss": d_loss,
                "disc_pos": pos,
                "disc_neg": neg,
            }
        )
        return out

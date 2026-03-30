# deepfillv2_ops_pt.py
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


IntOrPair = Union[int, Tuple[int, int]]
Activation = Optional[Callable[[torch.Tensor], torch.Tensor]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def tf_same_pad_2d(
    x: torch.Tensor,
    kernel_size: IntOrPair,
    stride: IntOrPair = 1,
    dilation: IntOrPair = 1,
) -> torch.Tensor:
    """
    TensorFlow-style SAME padding for NCHW tensors.
    This matches tf.layers.conv2d(..., padding='SAME').
    """
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)

    h, w = x.shape[-2:]
    out_h = math.ceil(h / sh)
    out_w = math.ceil(w / sw)

    eff_kh = (kh - 1) * dh + 1
    eff_kw = (kw - 1) * dw + 1

    pad_h_total = max((out_h - 1) * sh + eff_kh - h, 0)
    pad_w_total = max((out_w - 1) * sw + eff_kw - w, 0)

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    if pad_h_total == 0 and pad_w_total == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def _symmetric_indices(length: int, left: int, right: int, device: torch.device) -> torch.Tensor:
    """
    TensorFlow SYMMETRIC padding indices for one dimension.
    Example for length=4:
      original indices:  [0, 1, 2, 3]
      symmetric padded:  ... 1, 0, 0, 1, 2, 3, 3, 2 ...
    """
    positions = torch.arange(-left, length + right, device=device, dtype=torch.long)
    period = 2 * length
    positions = torch.remainder(positions, period)
    return torch.where(positions < length, positions, period - 1 - positions)


def symmetric_pad_2d(x: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    TensorFlow-style SYMMETRIC pad for NCHW tensors.
    padding = (left, right, top, bottom)
    """
    left, right, top, bottom = padding
    if left == right == top == bottom == 0:
        return x

    h_idx = _symmetric_indices(x.shape[-2], top, bottom, x.device)
    w_idx = _symmetric_indices(x.shape[-1], left, right, x.device)

    x = x.index_select(-2, h_idx)
    x = x.index_select(-1, w_idx)
    return x

def resize_nearest_tf_align_corners(x: torch.Tensor, to_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Approximate TF1 resize_nearest_neighbor(..., align_corners=True) for NCHW tensors.
    This matters for the downscale path inside contextual_attention.
    """
    out_h, out_w = to_shape
    in_h, in_w = x.shape[-2:]

    if (in_h, in_w) == (out_h, out_w):
        return x

    def make_idx(in_size: int, out_size: int, device: torch.device) -> torch.Tensor:
        if out_size <= 1:
            return torch.zeros(out_size, dtype=torch.long, device=device)
        return torch.round(
            torch.linspace(0, in_size - 1, out_size, device=device)
        ).long()

    h_idx = make_idx(in_h, out_h, x.device)
    w_idx = make_idx(in_w, out_w, x.device)

    return x.index_select(-2, h_idx).index_select(-1, w_idx)

def resize_nearest(
    x: torch.Tensor,
    scale: Optional[float] = None,
    to_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if to_shape is None:
        if scale is None:
            raise ValueError("Either scale or to_shape must be provided.")
        return F.interpolate(x, scale_factor=scale, mode="nearest")
    return F.interpolate(x, size=to_shape, mode="nearest")


def resize_bilinear(
    x: torch.Tensor,
    scale: Optional[float] = None,
    to_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if to_shape is None:
        if scale is None:
            raise ValueError("Either scale or to_shape must be provided.")
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True)
    return F.interpolate(x, size=to_shape, mode="bilinear", align_corners=True)


#def resize_mask_like(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#    return resize_nearest(mask, to_shape=x.shape[-2:])
def resize_mask_like(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return resize_nearest_tf_align_corners(mask, x.shape[-2:])

class GenConv2d(nn.Module):
    """
    PyTorch port of inpaint_ops.gen_conv.

    Important:
      - `cnum` matches the TensorFlow call argument.
      - For non-output layers, the raw conv output is split in half:
            feature, gate = split(conv(x), 2, channel)
        then output = activation(feature) * sigmoid(gate)
      - Therefore effective output channels are:
            cnum // 2   if cnum != 3 and activation is not None
            cnum        otherwise
    """

    def __init__(
        self,
        in_channels: int,
        cnum: int,
        ksize: IntOrPair,
        stride: IntOrPair = 1,
        rate: IntOrPair = 1,
        padding: str = "SAME",
        activation: Activation = F.elu,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.cnum = cnum
        self.ksize = _pair(ksize)
        self.stride = _pair(stride)
        self.rate = _pair(rate)
        self.padding_mode = padding.upper()
        self.activation = activation

        if self.cnum != 3 and self.activation is not None and self.cnum % 2 != 0:
            raise ValueError(f"cnum must be even for gated split, got {self.cnum}.")

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnum,
            kernel_size=self.ksize,
            stride=self.stride,
            dilation=self.rate,
            padding=0,
            bias=bias,
        )

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "SAME":
            return tf_same_pad_2d(x, self.ksize, self.stride, self.rate)

        if self.padding_mode in ("SYMMETRIC", "REFLECT"):
            kh, kw = self.ksize
            dh, dw = self.rate
            ph = int(dh * (kh - 1) / 2)
            pw = int(dw * (kw - 1) / 2)
            if self.padding_mode == "SYMMETRIC":
                return symmetric_pad_2d(x, (pw, pw, ph, ph))
            return F.pad(x, (pw, pw, ph, ph), mode="reflect")

        raise ValueError(f"Unsupported padding mode: {self.padding_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_input(x)
        x = self.conv(x)

        if self.cnum == 3 or self.activation is None:
            return x

        feat, gate = torch.chunk(x, 2, dim=1)
        feat = self.activation(feat)
        gate = torch.sigmoid(gate)
        return feat * gate


class GenDeconv2d(nn.Module):
    """
    PyTorch port of inpaint_ops.gen_deconv:
      resize_nearest(x, scale=2) -> gen_conv(..., ksize=3, stride=1)
    """

    def __init__(
        self,
        in_channels: int,
        cnum: int,
        padding: str = "SAME",
        activation: Activation = F.elu,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = GenConv2d(
            in_channels=in_channels,
            cnum=cnum,
            ksize=3,
            stride=1,
            rate=1,
            padding=padding,
            activation=activation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_nearest(x, scale=2.0)
        return self.conv(x)


def _l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (torch.sum(x ** 2).sqrt() + eps)


class SNConv2d(nn.Module):
    """
    PyTorch port of conv2d_spectral_norm + kernel_spectral_norm.

    Notes:
      - weight is normalized every forward pass
      - one power iteration
      - u is persistent and non-trainable
      - normalization matches the TensorFlow code's matrix shape convention:
            reshape(kernel, [-1, out_channels])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: str = "VALID",
        dilation: IntOrPair = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding_mode = padding.upper()
        self.dilation = _pair(dilation)

        kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # This follows the TF code conceptually: a persistent non-trainable u.
        self.register_buffer("u", torch.randn(1, out_channels))

        # I am not pinning the exact TF default conv initializer here yet.
        # Leave PyTorch defaults out of this first structural port.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _normalized_weight(self) -> torch.Tensor:
        # Match TF's reshape(kernel, [-1, out_channels]).
        w_mat = self.weight.view(self.out_channels, -1).t()  # [N, Cout]

        # Important: do not use self.u directly in the autograd graph and then
        # update it in-place during the same forward pass. That triggers PyTorch
        # version-counter errors during backward. The buffer is non-trainable, so
        # we detach+clone it for the power iteration, then update the persistent
        # buffer under no_grad after computing the normalized weight.
        u = self.u.detach().clone()
        v_hat = _l2_norm(u @ w_mat.t())
        u_hat = _l2_norm(v_hat @ w_mat)
        sigma = v_hat @ w_mat @ u_hat.t()  # [1,1]

        w_norm = w_mat / sigma

        # TF updates u through a control dependency every run.
        with torch.no_grad():
            self.u.copy_(u_hat.detach())

        return w_norm.t().contiguous().view_as(self.weight)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "SAME":
            return tf_same_pad_2d(x, self.kernel_size, self.stride, self.dilation)
        if self.padding_mode == "VALID":
            return x
        raise ValueError(f"Unsupported padding mode for SNConv2d: {self.padding_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_input(x)
        weight = self._normalized_weight()
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )


class DisConv2d(nn.Module):
    """
    PyTorch port of inpaint_ops.dis_conv:
      conv2d_spectral_norm(..., ksize=5, stride=2, padding='SAME')
      -> leaky_relu
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 5,
        stride: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = SNConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding="SAME",
            dilation=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.leaky_relu(x, negative_slope=0.2)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x, start_dim=1)


def gan_hinge_loss(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matches neuralgym.gan_ops.gan_hinge_loss.

    Returns:
        g_loss, d_loss
    """
    hinge_pos = torch.mean(F.relu(1.0 - pos))
    hinge_neg = torch.mean(F.relu(1.0 + neg))
    d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg
    g_loss = -torch.mean(neg)
    return g_loss, d_loss


class SNPatchDiscriminator(nn.Module):
    """
    PyTorch port of build_sn_patch_gan_discriminator.

    Input channels depend on how you build the GAN input:
      - image only: 3
      - image + mask: 4
      - image + mask + edge: 5
    In the provided default config, gan_with_mask=True and guided=False,
    so the default discriminator input is 4 channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        cnum = 64
        self.conv1 = DisConv2d(in_channels, cnum)
        self.conv2 = DisConv2d(cnum, cnum * 2)
        self.conv3 = DisConv2d(cnum * 2, cnum * 4)
        self.conv4 = DisConv2d(cnum * 4, cnum * 4)
        self.conv5 = DisConv2d(cnum * 4, cnum * 4)
        self.conv6 = DisConv2d(cnum * 4, cnum * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return flatten(x)

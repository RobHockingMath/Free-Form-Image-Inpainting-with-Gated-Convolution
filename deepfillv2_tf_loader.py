# deepfillv2_tf_loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from deepfillv2_model_pt import DeepFillV2Model
from deepfillv2_ops_pt import SNConv2d


@dataclass
class TFLoadReport:
    checkpoint_path: str
    loaded_tf_names: List[str]
    missing_expected_tf_names: List[str]
    unused_model_tf_names: List[str]


def _resolve_tf_checkpoint_path(path: str) -> str:
    """
    Accept either:
      - a checkpoint prefix path
      - a directory containing checkpoint files + 'checkpoint' metadata
    """
    import tensorflow as tf

    if os.path.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"No TensorFlow checkpoint found in directory: {path}")
        return ckpt
    return path


def _make_tf_reader(path: str):
    import tensorflow as tf

    ckpt = _resolve_tf_checkpoint_path(path)
    reader = tf.train.load_checkpoint(ckpt)
    return ckpt, reader


def _get_tf_var_map(reader) -> Dict[str, List[int]]:
    return dict(reader.get_variable_to_shape_map())


def _get_tensor(reader, name: str) -> np.ndarray:
    return reader.get_tensor(name)


def _find_first_existing_name(var_map: Dict[str, List[int]], candidates: List[str]) -> str:
    for name in candidates:
        if name in var_map:
            return name
    raise KeyError(f"None of these TensorFlow variable names exist: {candidates}")


def _tf_conv_kernel_to_torch(kernel_hwio: np.ndarray) -> torch.Tensor:
    """
    TensorFlow conv kernel: [KH, KW, Cin, Cout]
    PyTorch conv kernel:   [Cout, Cin, KH, KW]
    """
    if kernel_hwio.ndim != 4:
        raise ValueError(f"Expected 4D conv kernel, got shape {kernel_hwio.shape}")
    return torch.from_numpy(kernel_hwio).permute(3, 2, 0, 1).contiguous()


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError(
            f"Shape mismatch for {name}: destination {tuple(dst.shape)} vs source {tuple(src.shape)}"
        )
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _load_plain_conv_from_tf(
    conv: nn.Conv2d,
    reader,
    var_map: Dict[str, List[int]],
    tf_prefix: str,
    consumed: Set[str],
) -> None:
    kernel_name = _find_first_existing_name(var_map, [f"{tf_prefix}/kernel", f"{tf_prefix}/weights"])
    bias_name = _find_first_existing_name(var_map, [f"{tf_prefix}/bias", f"{tf_prefix}/biases"])

    kernel = _get_tensor(reader, kernel_name)
    bias = _get_tensor(reader, bias_name)

    weight_t = _tf_conv_kernel_to_torch(kernel)
    bias_t = torch.from_numpy(bias).contiguous()

    with torch.no_grad():
        _copy_tensor(conv.weight, weight_t, f"{tf_prefix}.weight")
        if conv.bias is None:
            raise ValueError(f"{tf_prefix} has a TF bias but destination conv has no bias parameter")
        _copy_tensor(conv.bias, bias_t, f"{tf_prefix}.bias")

    consumed.add(kernel_name)
    consumed.add(bias_name)


def _load_snconv_from_tf(
    snconv: SNConv2d,
    reader,
    var_map: Dict[str, List[int]],
    tf_prefix: str,
    consumed: Set[str],
) -> None:
    kernel_name = _find_first_existing_name(var_map, [f"{tf_prefix}/kernel", f"{tf_prefix}/weights"])
    bias_name = _find_first_existing_name(var_map, [f"{tf_prefix}/bias", f"{tf_prefix}/biases"])
    u_name = _find_first_existing_name(var_map, [f"{tf_prefix}/kernel_sn/u"])

    kernel = _get_tensor(reader, kernel_name)
    bias = _get_tensor(reader, bias_name)
    u = _get_tensor(reader, u_name)

    weight_t = _tf_conv_kernel_to_torch(kernel)
    bias_t = torch.from_numpy(bias).contiguous()
    u_t = torch.from_numpy(u).contiguous()

    with torch.no_grad():
        _copy_tensor(snconv.weight, weight_t, f"{tf_prefix}.weight")
        if snconv.bias is None:
            raise ValueError(f"{tf_prefix} has a TF bias but destination SNConv2d has no bias parameter")
        _copy_tensor(snconv.bias, bias_t, f"{tf_prefix}.bias")
        _copy_tensor(snconv.u, u_t, f"{tf_prefix}.u")

    consumed.add(kernel_name)
    consumed.add(bias_name)
    consumed.add(u_name)


def _generator_target_map(model: DeepFillV2Model) -> Dict[str, nn.Module]:
    g = model.generator
    return {
        # stage 1
        "inpaint_net/conv1": g.conv1.conv,
        "inpaint_net/conv2_downsample": g.conv2_downsample.conv,
        "inpaint_net/conv3": g.conv3.conv,
        "inpaint_net/conv4_downsample": g.conv4_downsample.conv,
        "inpaint_net/conv5": g.conv5.conv,
        "inpaint_net/conv6": g.conv6.conv,
        "inpaint_net/conv7_atrous": g.conv7_atrous.conv,
        "inpaint_net/conv8_atrous": g.conv8_atrous.conv,
        "inpaint_net/conv9_atrous": g.conv9_atrous.conv,
        "inpaint_net/conv10_atrous": g.conv10_atrous.conv,
        "inpaint_net/conv11": g.conv11.conv,
        "inpaint_net/conv12": g.conv12.conv,
        "inpaint_net/conv13_upsample/conv13_upsample_conv": g.conv13_upsample.conv.conv,
        "inpaint_net/conv14": g.conv14.conv,
        "inpaint_net/conv15_upsample/conv15_upsample_conv": g.conv15_upsample.conv.conv,
        "inpaint_net/conv16": g.conv16.conv,
        "inpaint_net/conv17": g.conv17.conv,
        # stage 2 conv branch
        "inpaint_net/xconv1": g.xconv1.conv,
        "inpaint_net/xconv2_downsample": g.xconv2_downsample.conv,
        "inpaint_net/xconv3": g.xconv3.conv,
        "inpaint_net/xconv4_downsample": g.xconv4_downsample.conv,
        "inpaint_net/xconv5": g.xconv5.conv,
        "inpaint_net/xconv6": g.xconv6.conv,
        "inpaint_net/xconv7_atrous": g.xconv7_atrous.conv,
        "inpaint_net/xconv8_atrous": g.xconv8_atrous.conv,
        "inpaint_net/xconv9_atrous": g.xconv9_atrous.conv,
        "inpaint_net/xconv10_atrous": g.xconv10_atrous.conv,
        # stage 2 attention branch
        "inpaint_net/pmconv1": g.pmconv1.conv,
        "inpaint_net/pmconv2_downsample": g.pmconv2_downsample.conv,
        "inpaint_net/pmconv3": g.pmconv3.conv,
        "inpaint_net/pmconv4_downsample": g.pmconv4_downsample.conv,
        "inpaint_net/pmconv5": g.pmconv5.conv,
        "inpaint_net/pmconv6": g.pmconv6.conv,
        "inpaint_net/pmconv9": g.pmconv9.conv,
        "inpaint_net/pmconv10": g.pmconv10.conv,
        # stage 2 merge
        "inpaint_net/allconv11": g.allconv11.conv,
        "inpaint_net/allconv12": g.allconv12.conv,
        "inpaint_net/allconv13_upsample/allconv13_upsample_conv": g.allconv13_upsample.conv.conv,
        "inpaint_net/allconv14": g.allconv14.conv,
        "inpaint_net/allconv15_upsample/allconv15_upsample_conv": g.allconv15_upsample.conv.conv,
        "inpaint_net/allconv16": g.allconv16.conv,
        "inpaint_net/allconv17": g.allconv17.conv,
    }


def _discriminator_target_map(model: DeepFillV2Model) -> Dict[str, SNConv2d]:
    d = model.discriminator
    return {
        "discriminator/sn_patch_gan/conv1": d.conv1.conv,
        "discriminator/sn_patch_gan/conv2": d.conv2.conv,
        "discriminator/sn_patch_gan/conv3": d.conv3.conv,
        "discriminator/sn_patch_gan/conv4": d.conv4.conv,
        "discriminator/sn_patch_gan/conv5": d.conv5.conv,
        "discriminator/sn_patch_gan/conv6": d.conv6.conv,
    }


def expected_tf_model_variable_names(
    model: DeepFillV2Model,
    include_generator: bool = True,
    include_discriminator: bool = True,
) -> List[str]:
    names: List[str] = []

    if include_generator:
        for prefix in _generator_target_map(model).keys():
            names.extend([f"{prefix}/kernel", f"{prefix}/bias"])

    if include_discriminator:
        for prefix in _discriminator_target_map(model).keys():
            names.extend([f"{prefix}/kernel", f"{prefix}/bias", f"{prefix}/kernel_sn/u"])

    return names


def load_deepfillv2_from_tf_checkpoint(
    model: DeepFillV2Model,
    checkpoint_path: str,
    load_generator: bool = True,
    load_discriminator: bool = True,
) -> TFLoadReport:
    """
    Load visible model weights from a TensorFlow v1 checkpoint into the PyTorch model.

    This consumes:
      - generator conv kernels/biases under inpaint_net/...
      - discriminator SN conv kernels/biases/u under discriminator/sn_patch_gan/...

    It intentionally ignores optimizer slots and unrelated variables.
    """
    ckpt, reader = _make_tf_reader(checkpoint_path)
    var_map = _get_tf_var_map(reader)
    consumed: Set[str] = set()

    with torch.no_grad():
        if load_generator:
            for tf_prefix, conv in _generator_target_map(model).items():
                _load_plain_conv_from_tf(conv, reader, var_map, tf_prefix, consumed)

        if load_discriminator:
            for tf_prefix, snconv in _discriminator_target_map(model).items():
                _load_snconv_from_tf(snconv, reader, var_map, tf_prefix, consumed)

    expected_names = expected_tf_model_variable_names(
        model,
        include_generator=load_generator,
        include_discriminator=load_discriminator,
    )

    missing = [name for name in expected_names if name not in consumed]
    model_prefixes = ("inpaint_net/", "discriminator/")
    unused_model_tf_names = sorted(
        name for name in var_map.keys()
        if name.startswith(model_prefixes) and name not in consumed
    )

    return TFLoadReport(
        checkpoint_path=ckpt,
        loaded_tf_names=sorted(consumed),
        missing_expected_tf_names=sorted(missing),
        unused_model_tf_names=unused_model_tf_names,
    )


def save_pytorch_checkpoint(model: DeepFillV2Model, path: str) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


if __name__ == "__main__":
    # Example usage:
    #
    #   python deepfillv2_tf_loader.py \
    #       --tf_ckpt model_logs/release_places2_256 \
    #       --out deepfillv2_places2.pt
    #
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_ckpt", type=str, required=True,
                        help="TensorFlow checkpoint prefix or directory")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .pt file")
    parser.add_argument("--guided", action="store_true",
                        help="Use the guided 5-channel model (RGB+mask+edge for server input path)")
    parser.add_argument("--no_discriminator", action="store_true",
                        help="Load generator only")
    parser.add_argument("--cpu", action="store_true",
                        help="Force model to stay on CPU")
    args = parser.parse_args()

    # Match the visible default training/inference config unless overridden.
    model = DeepFillV2Model(
        guided=args.guided,
        gan_with_mask=True,
        edge_threshold=0.6,
        padding="SAME",
        gan_loss_alpha=1.0,
        l1_loss_alpha=1.0,
        ae_loss=True,
    )

    if not args.cpu and torch.cuda.is_available():
        model = model.cuda()

    report = load_deepfillv2_from_tf_checkpoint(
        model,
        checkpoint_path=args.tf_ckpt,
        load_generator=True,
        load_discriminator=not args.no_discriminator,
    )

    print(f"Resolved TensorFlow checkpoint: {report.checkpoint_path}")
    print(f"Loaded TF tensors: {len(report.loaded_tf_names)}")

    if report.missing_expected_tf_names:
        print("\nMissing expected TF tensors:")
        for name in report.missing_expected_tf_names:
            print(f"  {name}")

    if report.unused_model_tf_names:
        print("\nUnused TF tensors under model prefixes:")
        for name in report.unused_model_tf_names:
            print(f"  {name}")

    save_pytorch_checkpoint(model, args.out)
    print(f"\nSaved PyTorch checkpoint to: {args.out}")
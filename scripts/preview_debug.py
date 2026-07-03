"""Step-by-step pipeline debug previews (used by debug_pipeline / run_pipeline)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from head_align.axial_detector import pca_axis_segments
from head_align.inference import center_slice_np
from utils import apply_brain_ct_window

PLANES = ("axial", "coronal", "sagittal")
PLANE_LABEL = {"axial": "axial (Z)", "coronal": "coronal (Y)", "sagittal": "sagittal (X)"}


def _slice_windowed(vol_zyx: np.ndarray, plane: str) -> np.ndarray:
    return apply_brain_ct_window(center_slice_np(vol_zyx, plane))


def _draw_bottom_slab_overlay(ax: plt.Axes, vol_zyx: np.ndarray, *, n_keep: int) -> None:
    z = int(vol_zyx.shape[0])
    n_keep = int(n_keep)
    if n_keep <= 0 or z <= n_keep:
        return
    z_discard = z - n_keep
    w = int(ax.get_images()[0].get_array().shape[1])
    ax.add_patch(
        Rectangle((0, 0), w, z_discard, linewidth=0, edgecolor="none", facecolor="#888888", alpha=0.45)
    )
    ax.add_patch(
        Rectangle((0, z_discard), w, n_keep, linewidth=0, edgecolor="none", facecolor="#33cc66", alpha=0.30)
    )
    ax.axhline(z_discard, color="#33cc66", linewidth=1.2, linestyle="--")


def _draw_pca_guidelines(
    ax: plt.Axes,
    *,
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    shape_yx: tuple[int, int],
) -> None:
    seg1, seg2 = pca_axis_segments(center_xy, e1, e2, shape_yx)
    (xa0, ya0), (xa1, ya1) = seg1
    (xb0, yb0), (xb1, yb1) = seg2
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ax.plot([xa0, xa1], [ya0, ya1], color="#ff6b35", linewidth=1.4, alpha=0.95)
    ax.plot([xb0, xb1], [yb0, yb1], color="#4ecdc4", linewidth=1.2, alpha=0.95)
    ax.plot(cx, cy, marker="+", markersize=9, markeredgewidth=1.4, color="#ffe66d", linestyle="none")


def save_step_compare_preview(
    out_path: Path,
    *,
    case_id: str,
    step_name: str,
    step_desc: str,
    vol_before: np.ndarray,
    vol_after: np.ndarray,
    overlay_before: str | None = None,
    overlay_kwargs: dict | None = None,
    dpi: int = 120,
) -> None:
    overlay_kwargs = overlay_kwargs or {}
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.08)

    for col, plane in enumerate(PLANES):
        for row_idx, (label, vol) in enumerate(
            (("исходный", vol_before), (f"шаг: {step_name}", vol_after))
        ):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(
                _slice_windowed(np.asarray(vol, dtype=np.float32), plane),
                cmap="gray",
                origin="upper",
                aspect="equal",
            )
            if row_idx == 0 and overlay_before == "bottom_slab" and plane in ("coronal", "sagittal"):
                _draw_bottom_slab_overlay(ax, vol_before, **overlay_kwargs)
            if row_idx == 0 and overlay_before == "pca" and plane == "axial":
                img_shape = ax.get_images()[0].get_array().shape
                _draw_pca_guidelines(
                    ax,
                    center_xy=np.asarray(overlay_kwargs["center_xy"], dtype=np.float64),
                    e1=np.asarray(overlay_kwargs["e1"], dtype=np.float64),
                    e2=np.asarray(overlay_kwargs["e2"], dtype=np.float64),
                    shape_yx=img_shape,
                )
            if row_idx == 1 and overlay_before == "pca" and plane == "axial":
                img_shape = ax.get_images()[0].get_array().shape
                cx, cy = img_shape[1] / 2, img_shape[0] / 2
                ax.plot(
                    cx,
                    cy,
                    marker="o",
                    markersize=5,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    color="#ffe66d",
                )
            ax.set_title(f"{label}\n{PLANE_LABEL[plane]}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"{case_id} | {step_desc}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_step3_final_preview(
    out_path: Path,
    *,
    case_id: str,
    step_desc: str,
    vol_slab: np.ndarray,
    vol_axial_aligned: np.ndarray,
    vol_pose_in: np.ndarray,
    vol_pose_out: np.ndarray,
    pca_overlay: dict,
    dpi: int = 120,
) -> None:
    fig = plt.figure(figsize=(12, 13))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.08)

    rows = (
        ("исходный", vol_slab, "orig"),
        ("axial PCA", vol_axial_aligned, "ax_aligned"),
        ("pose in", vol_pose_in, "pose_in"),
        ("pose out", vol_pose_out, "pose_out"),
    )

    for row_idx, (label, vol, mode) in enumerate(rows):
        for col, plane in enumerate(PLANES):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(
                _slice_windowed(np.asarray(vol, dtype=np.float32), plane),
                cmap="gray",
                origin="upper",
                aspect="equal",
            )
            if mode == "orig" and plane == "axial":
                img_shape = ax.get_images()[0].get_array().shape
                _draw_pca_guidelines(
                    ax,
                    center_xy=np.asarray(pca_overlay["center_xy"], dtype=np.float64),
                    e1=np.asarray(pca_overlay["e1"], dtype=np.float64),
                    e2=np.asarray(pca_overlay["e2"], dtype=np.float64),
                    shape_yx=img_shape,
                )
            if mode == "ax_aligned" and plane == "axial":
                img_shape = ax.get_images()[0].get_array().shape
                cx, cy = img_shape[1] / 2, img_shape[0] / 2
                ax.plot(
                    cx,
                    cy,
                    marker="o",
                    markersize=5,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    color="#ffe66d",
                )
            ax.set_title(f"{label}\n{PLANE_LABEL[plane]}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"{case_id} | {step_desc}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

"""Plot-only shared views / 只组织图像，不改变场、损耗或角度统计。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from pcselsim.fabrication import inclusion_boundary


def plot_device_overview(cell, lattice_nm: float, half_um: float, shape: str,
                         path: Path, direct=None) -> None:
    """Physical boundary + explicitly decimated centers + true-scale hole zoom.

    A 300-um device contains ~10^6 cells: subpixel holes in the overview are not
    enlarged and passed off as physical dimensions. The middle/right views show
    the actual motif. / 全貌中心点抽样标明；孔尺寸只在真实比例放大窗显示。
    """
    direct = np.eye(2) if direct is None else np.asarray(direct)
    a_um = lattice_nm * 1e-3
    primitive_area = abs(np.linalg.det(direct)) * a_um**2
    area = {"square": 4*half_um**2, "circle": np.pi*half_um**2,
            "hexagon": 1.5*np.sqrt(3)*half_um**2}[shape]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    ax = axes[0]
    if shape == "square":
        boundary = Rectangle((-half_um, -half_um), 2*half_um, 2*half_um)
    elif shape == "circle":
        boundary = Circle((0, 0), half_um)
    else:
        phase = np.arange(6)*np.pi/3
        boundary = Polygon(half_um*np.column_stack((np.cos(phase), np.sin(phase))))
    boundary.set(facecolor="#e3edf4", edgecolor="#174a7e", linewidth=2)
    ax.add_patch(boundary)
    extent = int(np.ceil(2*half_um/a_um))
    stride = max(1, int(np.ceil(extent/25)))
    order = np.arange(-extent, extent+1, stride)
    ii, jj = np.meshgrid(order, order)
    centers = np.column_stack((ii.ravel(), jj.ravel())) @ direct.T * a_um
    inside = boundary.get_path().contains_points(centers, boundary.get_patch_transform())
    ax.scatter(*centers[inside].T, s=1.3, c="#354d61", alpha=.6)
    bar = half_um/3
    ax.plot((-.85*half_um, -.85*half_um+bar), (-.88*half_um, -.88*half_um),
            color="black", lw=3)
    ax.text(-.85*half_um+bar/2, -.82*half_um, f"{bar:g} um", ha="center")
    ax.set(aspect="equal", xlim=(-1.1*half_um, 1.1*half_um),
           ylim=(-1.1*half_um, 1.1*half_um), xlabel="x (um)", ylabel="y (um)",
           title=f"whole {shape} device: {2*half_um:g} um bounding width")
    ax.text(0, 1.01*half_um, f"~{area/primitive_area:,.0f} cells; centers every {stride} cells",
            ha="center", fontsize=8)
    boundaries = [inclusion_boundary(hole, direct)*lattice_nm for hole in cell.inclusions]
    for i in range(-3, 4):
        for j in range(-3, 4):
            offset = direct @ np.array((i, j))*lattice_nm
            for vertices in boundaries:
                axes[1].add_patch(Polygon(vertices+offset, facecolor="white", edgecolor="black", lw=.6))
    axes[1].set(aspect="equal", xlim=(-3*lattice_nm, 3*lattice_nm),
                ylim=(-3*lattice_nm, 3*lattice_nm), xlabel="x (nm)", ylabel="y (nm)",
                title="actual hole array: lattice zoom")
    primitive = np.array(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))) @ direct.T*lattice_nm
    for ax in axes[1:]:
        ax.add_patch(Polygon(primitive, fill=False, edgecolor="#d62728", lw=1.7))
    for index, vertices in enumerate(boundaries):
        axes[2].add_patch(Polygon(vertices, facecolor="#f7f7f7", edgecolor="black", lw=1.3))
        axes[2].text(*vertices.mean(axis=0), str(index+1), ha="center", va="center", fontsize=8)
    motif_extent = np.max(np.abs(np.vstack([primitive, *boundaries])), axis=0)*1.12
    axes[2].set(aspect="equal", xlim=(-motif_extent[0], motif_extent[0]),
                ylim=(-motif_extent[1], motif_extent[1]), xlabel="x (nm)", ylabel="y (nm)",
                title=f"selected motif: a={lattice_nm:.2f} nm")
    fig.suptitle("Device boundary and real-scale lattice / full-view center sampling is explicit")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_far_field_zoom(angle_deg, power, half_view_deg: float, path: Path, mode_name: str) -> None:
    """Crop a display of already evaluated FFP; never recompute RMS/energy here."""
    if half_view_deg <= 0:
        raise ValueError("FAR_FIELD_ZOOM_DEG must be positive")
    extent = (angle_deg[0], angle_deg[-1], angle_deg[0], angle_deg[-1])
    limit = min(half_view_deg, float(np.max(np.abs(angle_deg))))
    fig, ax = plt.subplots(figsize=(6, 5))
    artist = ax.imshow(power, origin="lower", extent=extent, cmap="hot", vmin=0,
                       vmax=max(float(np.max(power)), np.finfo(float).eps))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="theta_x (deg)",
           ylabel="theta_y (deg)", title=f"{mode_name}: display-only FFP zoom (+/-{limit:g} deg)")
    fig.colorbar(artist, ax=ax, label="relative intensity (original normalization)")
    fig.text(.5, .01, "Plot crop only; divergence and encircled energy retain the original evaluated window.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1))
    fig.savefig(path, dpi=200)
    plt.close(fig)

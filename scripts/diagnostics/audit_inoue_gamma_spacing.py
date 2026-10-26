"""Recompute the compact Inoue Gamma-spacing audit without changing source data.

Run from any directory with the project's NumPy/SciPy environment::

    python -B scripts/diagnostics/audit_inoue_gamma_spacing.py

Only one small JSON is written. No finite-device solves, scans, plots, or large
arrays are produced. Paper values are digitized from the published figure and
are explicitly not the authors' original numerical data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack


SOURCE = ROOT / "results/current/20261021_paper_corrected_cold_r2"
OUTPUT = ROOT / "results/current/20261021_inoue_gamma_spacing_audit/audit.json"
PAPER_DIRECTORY = Path("C:/Users/19254/Zotero/storage/4WEC2TRI")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(data: dict, key: str) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.shape != (4, 4, 2):
        raise ValueError(f"Unexpected matrix shape for {key}: {values.shape}")
    return values[..., 0] + 1j * values[..., 1]


def summarize(coupling: np.ndarray, optical: dict, period_nm: float) -> dict:
    values = np.linalg.eigvals(coupling)
    values = values[np.lexsort((values.imag, values.real))]
    normalized = (
        period_nm / optical["wavelength_nm"]
        + values.real * period_nm * 1e-9 / (2 * np.pi * optical["group_index"])
    )
    gaps = np.diff(normalized)
    return {
        "labels": ["A", "B", "C", "D"],
        "label_definition": "Ascending real Gamma-point frequency; no paper eigenfield matching has been performed.",
        "detuning_cm^-1": (values.real / 100).tolist(),
        "amplitude_radiation_loss_cm^-1": (values.imag / 100).tolist(),
        "normalized_frequency_a_over_lambda": normalized.tolist(),
        "wavelength_nm": (period_nm / normalized).tolist(),
        "adjacent_labels": ["AB", "BC", "CD"],
        "adjacent_detuning_gaps_cm^-1": (np.diff(values.real) / 100).tolist(),
        "adjacent_normalized_frequency_gaps": gaps.tolist(),
        "BC_over_AB": float(gaps[1] / gaps[0]),
    }


def paper_digitization() -> dict:
    # PDF page 3, vector coordinates in points measured downwards from the top.
    # Axis ticks: .288 at y=298.216, .296 at y=221.530.  The filled Gamma
    # markers were extracted with pdfplumber from the original local PDF.
    marker_y = np.asarray([265.907, 262.737, 252.862, 250.790])
    frequency = .288 + (298.216 - marker_y) / (298.216 - 221.530) * .008
    gaps = np.diff(frequency)
    return {
        "doi": "10.1103/PhysRevB.99.035308",
        "url": "https://doi.org/10.1103/PhysRevB.99.035308",
        "location": "PDF page 3, Fig. 2(c)",
        "data_status": "Approximate figure digitization, NOT author-provided numerical arrays.",
        "method": "Linear calibration of vector-figure Gamma marker centres against the two frequency-axis tick positions; marker coordinates were read from the original PDF using pdfplumber.",
        "reproduction_note": "This script repeats the coordinate-to-frequency arithmetic from the recorded vector coordinates; it does not re-extract PDF paths.",
        "axis_tick_y_points": [298.216, 221.530],
        "axis_tick_frequencies": [.288, .296],
        "Gamma_marker_y_points_ABCD": marker_y.tolist(),
        "normalized_frequency_a_over_lambda_approximate": frequency.tolist(),
        "adjacent_normalized_frequency_gaps_approximate": gaps.tolist(),
        "BC_over_AB_approximate": float(gaps[1] / gaps[0]),
        "recommended_display_precision": "Gamma frequencies about 5 decimal places; gap ratio about 3.1. Extra digits only document the digitization arithmetic.",
        "geometry_evidence": "Page 3 text: actual embedded 3D holes were reconstructed from SEM cross sections and sliced along z to obtain average refractive index and coupling coefficients. The paper publishes the 0.25a x/y displacement, but not a complete numerical 3D contour or C matrix.",
    }


def main() -> None:
    inputs = [SOURCE / "parameters.json", SOURCE / "01_geometry_coupling.json"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    parameters = json.loads(inputs[0].read_text(encoding="utf-8"))
    saved = json.loads(inputs[1].read_text(encoding="utf-8"))
    lattice, optical = parameters["lattice"], parameters["optical"]
    period = lattice["constant_nm"]
    c1, cr, c2, total = [matrix(saved, k) for k in ["C1D", "Crad", "C2D", "C_total"]]
    if not np.allclose(total, c1 + cr + c2, rtol=1e-13, atol=1e-9):
        raise ValueError("The saved matrix components do not sum to C_total")
    current = summarize(total, optical, period)
    c2_within_xy = np.zeros((4, 4), dtype=complex)
    c2_within_xy[:2, :2] = c2[:2, :2]
    c2_within_xy[2:, 2:] = c2[2:, 2:]
    ablations = {
        "C1D_only": summarize(c1, optical, period),
        "C1D_plus_Crad": summarize(c1 + cr, optical, period),
        "C1D_plus_Crad_plus_C2D_diagonal": summarize(c1 + cr + np.diag(np.diag(c2)), optical, period),
        "C1D_plus_Crad_plus_C2D_within_xy_pairs": summarize(c1 + cr + c2_within_xy, optical, period),
        "full_C1D_plus_Crad_plus_C2D": current,
    }

    cell = SquareLatticeCell(
        lattice["background_index"] ** 2,
        tuple(Ellipse(
            (h["center_x_nm"] / period, h["center_y_nm"] / period),
            (h["radius_x_nm"] / period, h["radius_y_nm"] / period),
            angle_deg=h["rotation_deg"], epsilon=lattice["hole_index"] ** 2,
        ) for h in lattice["inclusions"]),
    )
    # Boundary indices/padding are the original paper-reference entry settings.
    stack = LayerStack(
        tuple(Layer(x["name"], x["thickness_nm"], x["refractive_index"])
              for x in parameters["layers"]),
        top_index=3.122, bottom_index=3.297, padding_um=1.0,
    )
    convergence = []
    baseline_error = None
    for order, dz in [(10, 3.), (30, 3.), (50, 3.), (10, 1.)]:
        result = build_geometry_coupling(
            cell, stack, "photonic crystal", period, optical["wavelength_nm"],
            ThreeDCWTSettings(truncation_order=order, vertical_step_nm=dz),
            fixed_wavelength_nm=optical["wavelength_nm"],
        )
        # Recomputed ng is reported; the matrix-gap comparison uses fixed
        # archived frequency conversion to isolate changes in C.
        entry = summarize(result.coupling_m, optical, period)
        entry.update({
            "D": order, "vertical_step_nm": dz,
            "recomputed_effective_index": result.effective_index,
            "recomputed_group_index": result.group_index,
            "relative_adjacent_gap_change_from_archived_D10_dz3": (
                np.asarray(entry["adjacent_detuning_gaps_cm^-1"])
                / np.asarray(current["adjacent_detuning_gaps_cm^-1"]) - 1
            ).tolist(),
        })
        convergence.append(entry)
        if order == 10 and dz == 3.:
            baseline_error = float(np.linalg.norm(result.coupling_m - total) / np.linalg.norm(total))
            if baseline_error > 1e-10:
                raise ValueError(f"Archived baseline was not reproduced: {baseline_error}")

    source_paths = [
        Path(__file__), ROOT / "src/pcselsim/three_d_cwt.py",
        ROOT / "src/pcselsim/band_structure.py", ROOT / "src/pcselsim/geometry.py",
        ROOT / "src/pcselsim/vertical.py", ROOT / "scripts/run_paper_reproduction.py",
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    pdfs = list(PAPER_DIRECTORY.glob("*.pdf")) if PAPER_DIRECTORY.exists() else []
    report = {
        "audit_label_date": "2026-10-21",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Small Gamma-matrix diagnostic only; original results are read-only. No finite-device or time-domain rerun, no large parameter scan.",
        "input_sha256": before, "code_sha256": source_hashes,
        "paper_source": {"path": str(pdfs[0]), "sha256": digest(pdfs[0])} if pdfs else {"status": "Local PDF unavailable; recorded vector coordinates retained."},
        "model_conditions": {
            "source_directory": str(SOURCE.relative_to(ROOT)),
            "lattice": lattice,
            "fill_fraction": parameters["derived_fill_fraction"],
            "vertical_layers": parameters["layers"],
            "top_cladding_index": 3.122, "bottom_cladding_index": 3.297,
            "vertical_padding_um": 1.,
            "fixed_wavelength_nm_for_convergence": optical["wavelength_nm"],
            "archived_effective_index": optical["effective_index"],
            "archived_group_index": optical["group_index"],
            "paper_group_index": 3.513,
            "archived_pc_confinement": saved["pc_confinement"],
            "archived_passivity_projection_correction_m^-1": saved["passivity_correction_m^-1"],
            "approximations": [
                "Two ideal ellipses with z-independent cross section through a 190 nm PC layer; not the published SEM-reconstructed 3D contour.",
                "Scalar TE0 vertical mode and wavelength-independent material indices.",
                "Interface-reflection-free Green function, four retained basic waves and truncated high-order sum.",
                "Frozen archived wavelength; no Bragg iteration or frequency-closure comparison in this convergence diagnostic.",
            ],
        },
        "paper_figure_digitization": paper_digitization(),
        "current_Gamma": current,
        "matrix_component_ablation": ablations,
        "convergence_recomputed_this_run": convergence,
        "baseline_matrix_relative_difference": baseline_error,
        "findings": {
            "established": [
                "The current adjacent gap ratio differs materially from the published figure; its Gamma gap pattern is not quantitatively reproduced.",
                "The current C2D inter-axis blocks change a close-pair / wide-central-gap spectrum into comparatively even gaps.",
                "A common frequency offset and common ng conversion cannot change the ratio of adjacent gaps.",
                "Increasing D through 50 retains the different gap pattern; refining dz from 3 to 1 nm changes the D10 gaps by less than 0.4%.",
                "D10 absolute gap values are not strictly converged: changes through D50 remain of order 10-16%.",
                "A-D labels here are ascending Gamma frequencies, without paper eigenfield matching.",
            ],
            "plausible_but_not_isolated_root_causes": [
                "Different lateral Fourier coefficients and z weighting because the actual SEM-derived geometry is not reconstructed.",
                "Limits of the present Green-function and scalar-mode approximations, or unverified high-order-coupling accuracy.",
            ],
            "not_established": [
                "This diagnostic does not prove a C2D implementation bug or identify a unique physical root cause.",
                "Ablation is not a validated replacement model; C2D must not be deleted to fit the picture.",
                "No experimental validation, full-Maxwell benchmark, or complete mode-symmetry correspondence is established.",
            ],
        },
    }
    after = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    if before != after:
        raise RuntimeError("Original input files changed during the audit")
    report["original_inputs_unchanged"] = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                      "current_BC_over_AB": current["BC_over_AB"],
                      "paper_BC_over_AB_approximate": report["paper_figure_digitization"]["BC_over_AB_approximate"],
                      "baseline_matrix_relative_difference": baseline_error,
                      "original_inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()

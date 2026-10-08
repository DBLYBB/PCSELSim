"""Entry orchestration only; no new optical equations / 流程与归档回归。"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_paper_reproduction as paper
from scripts import run_square_double_hole as square
from scripts import run_triangular_three_hole as triangle
from scripts._four_wave_workflow import build_unit_cell, threshold_current_audit
from scripts._run_controls import execute_run
from scripts._triangular_workflow import build_candidate, threshold_audit, write_threshold_audit
from pcselsim.config import load_config
from pcselsim.custom_analysis import LatticeSpec
from pcselsim.geometry import Ellipse
from pcselsim.triangular_six_wave import TriangularEllipse, TriangularPolygon

ROOT = Path(__file__).resolve().parents[1]


def test_root_has_exactly_three_semiconductor_main_scripts():
    mains = {path.name for path in (ROOT/"scripts").glob("run_*.py")}
    assert mains == {"run_paper_reproduction.py", "run_square_double_hole.py",
                     "run_triangular_three_hole.py", "run_ybyag_pcsel.py"}


def test_all_archived_scripts_import_without_running():
    archive = ROOT/"scripts"/"archive"/"legacy_20261008"
    for path in archive.glob("*.py"):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"scripts.archive.legacy_20261008.{path.stem}")
        assert callable(module.main)


@pytest.mark.parametrize("module", [paper, square, triangle])
def test_curated_main_runs_help_from_foreign_cwd(module, tmp_path):
    result = subprocess.run([sys.executable, str(Path(module.__file__)), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


def test_square_uses_two_smooth_circles_with_editable_nm_dimensions():
    p = square.build_parameters()
    lattice = LatticeSpec("square", p.LATTICE_CONSTANT_NM, p.HOLE_SHAPE,
                         p.HOLE_RADIUS_X_NM, p.HOLE_RADIUS_Y_NM, p.HOLE_ROTATION_DEG,
                         p.BACKGROUND_INDEX, p.HOLE_INDEX, p.UNIT_CELL_INCLUSIONS)
    cell = build_unit_cell(lattice)
    assert len(cell.inclusions) == 2
    assert all(isinstance(hole, Ellipse) and hole.radii[0] == hole.radii[1] for hole in cell.inclusions)
    assert np.isclose(lattice.fill_fraction, .14)
    assert p.THRESHOLD_CURRENT_SOURCE == "derived"


def test_triangle_uses_rounded_main_and_two_circles():
    p = triangle.build_parameters()
    candidate = build_candidate(p.DESIGN, p.HOLE_EPSILON)
    assert isinstance(candidate.inclusions[0], TriangularPolygon)
    assert all(isinstance(hole, TriangularEllipse) for hole in candidate.inclusions[1:])
    assert np.isclose(sum(hole.fill_fraction for hole in candidate.inclusions), .15)
    assert len(candidate.inclusions[0].vertices_fractional) > 3


def test_explicit_carrier_and_internal_loss_affect_triangle_threshold():
    p = triangle.build_parameters()
    baseline = threshold_audit(p, 100.0, (300e-6)**2, .044, 3.4)
    higher_loss = threshold_audit(replace(p, INTERNAL_LOSS_CM=10.0), 100.0, (300e-6)**2, .044, 3.4)
    assert higher_loss["required_modal_gain_cm-1"] == baseline["required_modal_gain_cm-1"] + 5
    assert higher_loss["threshold_current_A"] > baseline["threshold_current_A"]


def test_result_directory_refuses_overwrite(tmp_path):
    (tmp_path/"existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execute_run(square.build_parameters(), square.STEPS, "square", lambda *args: None,
                    ["--output", str(tmp_path)])
    assert (tmp_path/"existing.txt").read_text() == "keep"


def test_quick_uses_three_convergence_grids(tmp_path):
    observed = []
    execute_run(square.build_parameters(), square.STEPS, "square", lambda *args: observed.append(args),
                ["--quick", "--output", str(tmp_path/"new")])
    assert len(observed[0][0].FINITE_EIGEN_GRIDS) >= 3
    assert min(observed[0][0].FINITE_EIGEN_GRIDS) >= 9
    assert observed[0][0].END_TIME_NS < 1
    assert (tmp_path/"new"/"run_status.json").exists()


@pytest.mark.parametrize("module", [paper, square, triangle])
def test_curated_entries_expose_device_overview_and_plot_only_zoom(module):
    assert "02_device_overview" in module.STEPS
    assert any("far_field_zoom" in key for key in module.STEPS)
    assert module.build_parameters().FAR_FIELD_ZOOM_DEG > 0


def test_triangle_threshold_output_uses_explicit_inputs_without_global_modes(tmp_path):
    p = triangle.build_parameters()
    mode = SimpleNamespace(name="B1", alpha_per_m=100., extrapolation_status="grid_sensitive")
    far = SimpleNamespace(center_to_peak=1., peak_offset_deg=0., centroid_offset_deg=0.,
                          ellipticity=1., requested_view_deg=1., evaluated_view_deg=1.,
                          energy_normalization="evaluated_angular_window", encircled_power_0p5deg=.9,
                          encircled_power_1deg=1., full_rms_divergence_deg=.3)
    model = SimpleNamespace(bragg_wavelength_nm=935., group_index=3.5)
    audit = threshold_audit(p, mode.alpha_per_m, (300e-6)**2, .044, 3.4)
    output = write_threshold_audit(p, tmp_path, mode, far, .05, audit, model)
    assert output["far_field"]["selected_family_extrapolation_status"] == "grid_sensitive"
    assert (tmp_path/"11_threshold_current_audit.json").exists()


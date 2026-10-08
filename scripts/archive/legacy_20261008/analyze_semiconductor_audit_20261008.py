"""Read-only physics postprocessing of the dated semiconductor rerun.

No preset or model is changed. FFTs retain coherent DC and use the same final
2-ns Hann window as the main workflow. Multiple peaks in one spatial/modal
projection are diagnostics, not proof of a complete transverse-mode spectrum.
中文：读取原始NPZ，不重新拟合参数；保存谱数组、功率统计和源代码运行指纹。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import find_peaks

from pcselsim.constants import c
from pcselsim.observables import wavelength_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=PROJECT_ROOT / "results" / "custom_semiconductor_audit_20261008")
    args = parser.parse_args()
    directory = args.directory.resolve()
    threshold = json.loads((directory / "05_threshold_current_audit.json").read_text(encoding="utf-8"))
    params = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
    reference = threshold["spectrum_reference_wavelength_nm"]
    coupling = json.loads((directory / "01_geometry_coupling.json").read_text(encoding="utf-8"))
    encoded = np.asarray(coupling["band_eigenvectors"])
    basis = encoded[..., 0] + 1j * encoded[..., 1]
    gram_error = float(np.linalg.norm(basis.conj().T @ basis - np.identity(4)))
    rows = []
    diagnostics = []
    paths = sorted(path for path in directory.glob("08_I_*.npz")
                   if not path.stem.endswith("_spectrum_arrays"))
    if not paths:
        raise FileNotFoundError("No saved time-domain NPZ results exist")
    fig, axes = plt.subplots(len(paths), 1, figsize=(9, 2.4 * len(paths)), squeeze=False)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            ratio = float(data["current_ratio"])
            time = data["time_ns"]
            power = data["power_W"]
            carrier = data["center_carrier_cm3"]
            sample_dt = float(data["dt_s"])
            count = max(16, int(round(2e-9 / sample_dt)))
            signal = data["complex_signal"][-count:]
            wavelength, norm = wavelength_spectrum(signal, sample_dt, reference,
                                                   window=True, remove_mean=False)
            offsets = np.fft.fftfreq(signal.size, sample_dt)
            physical_frequency = c / (reference * 1e-9) + offsets
            raw = np.abs(np.fft.fft(signal * np.hanning(signal.size)))**2
            valid = physical_frequency > 0
            order = np.argsort(c / physical_frequency[valid] * 1e9)
            offsets = offsets[valid][order]
            raw = raw[valid][order]
            peaks, info = find_peaks(norm, height=.05, prominence=.04, distance=3)
            peaks = peaks[np.argsort(norm[peaks])[::-1]]
            peak_records = [{"wavelength_nm": float(wavelength[i]),
                             "relative_fft_power": float(norm[i]),
                             "frequency_offset_GHz": float(offsets[i] / 1e9)} for i in peaks[:20]]
            # Snapshot coefficients are not a complete nonlinear modal history.
            coefficients = np.linalg.solve(basis, data["final_field"].reshape(4, -1))
            weights = np.sum(np.abs(coefficients)**2, axis=1)
            weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
            steady = time >= time[-1] - 1.0
            first = (time >= time[-1] - 1.0) & (time < time[-1] - .5)
            second = time >= time[-1] - .5
            mean = float(np.mean(power[steady]))
            relative_drift = float((np.mean(power[second]) - np.mean(power[first])) /
                                   max(mean, np.finfo(float).tiny))
            row = {
                "current_ratio_paper_threshold": ratio,
                "current_A": ratio * threshold["selected_threshold_A"],
                "ratio_to_uniform_cold_cavity_threshold": ratio * threshold["selected_threshold_A"] /
                    threshold["derived_uniform_injection_threshold_A"],
                "end_time_ns": float(time[-1]), "sample_dt_fs": sample_dt * 1e15,
                "spectrum_window_ns": signal.size * sample_dt * 1e9,
                "frequency_bin_GHz": 1.0 / (signal.size * sample_dt) / 1e9,
                "mean_last_1ns_power_W": mean,
                "max_power_W": float(np.max(power)),
                "relative_power_std_last_1ns": float(np.std(power[steady]) / max(mean, np.finfo(float).tiny)),
                "relative_half_window_power_drift": relative_drift,
                "mean_last_1ns_center_carrier_cm-3": float(np.mean(carrier[steady])),
                "number_resolvable_A_projection_peaks_5percent": int(len(peaks)),
                "dominant_wavelength_nm": float(wavelength[np.argmax(norm)]),
                **{f"final_band_coefficient_fraction_{name}": float(weight)
                   for name, weight in zip("ABCD", weights, strict=True)},
                "all_arrays_finite": bool(all(np.all(np.isfinite(data[key])) for key in
                                               ("time_ns", "power_W", "complex_signal", "final_field", "final_carrier_cm3"))),
            }
            np.savez_compressed(directory / f"{path.stem}_spectrum_arrays.npz",
                                wavelength_nm=wavelength, normalized_fft_power=norm,
                                raw_fft_power_arbitrary_units=raw, frequency_offset_Hz=offsets,
                                complex_signal_window=signal, sample_dt_s=sample_dt,
                                reference_wavelength_nm=reference, remove_mean=False)
            rows.append(row)
            diagnostics.append({**row, "resolvable_peak_list": peak_records,
                                "solver_metadata": json.loads(str(data["metadata_json"]))})
            axis = axes[index, 0]
            # Display the represented optical-frequency interval; the original
            # ±0.3-nm plotting window can conceal distant band-edge peaks.
            visible = (wavelength > reference - 15) & (wavelength < reference + 15)
            axis.plot(wavelength[visible], norm[visible], lw=.8)
            axis.set(xlabel="Wavelength (nm)", ylabel="FFT power / peak",
                     title=f"I={row['current_A']:.3f} A, {ratio:.2f} x paper Ith; A projection, last 2 ns, DC retained")
            axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "09_full_span_spectra.png", dpi=180)
    plt.close(fig)
    with (directory / "09_postprocess_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__},
              "reference_wavelength_nm": reference,
              "threshold_reference_source": threshold["selected_threshold_source"],
              "cold_cavity_domain_um": params["optical"]["domain_um"],
              "time_domain_um": params["extra"]["device"]["domain_um"],
              "used_active_confinement": params["extra"]["used_active_confinement"],
              "solved_active_confinement": params["extra"]["solved_vertical_TE0_active_confinement"],
              "eigenbasis_gram_frobenius_error": gram_error,
              "runs": diagnostics,
              "scope": ["A-projected spatially integrated complex signal, not all modal spectra",
                        "Peak counts may include noise, chirp or modulation; not all are transverse eigenmodes",
                        "Final eigenbasis coefficient weights are snapshots, not lasing-mode competition proof",
                        "Raw FFT power has arbitrary units, not optical watts; power_W is from the field quadratic form",
                        "No linewidth calibration, thermal calculation, dx/dt convergence or full-mode spectrum"]}
    (directory / "09_postprocess_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = directory / "00_run_provenance.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        after = [{"path": item["path"], "sha256": hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()}
                 for item in manifest["source_sha256_before"]]
        manifest["source_sha256_after"] = after
        manifest["source_changed_during_run"] = [item["path"] for item, now in
                                                 zip(manifest["source_sha256_before"], after, strict=True)
                                                 if item["sha256"] != now["sha256"]]
        semiconductor_imported_names = {
            "run_custom_semiconductor_pcsel.py", "__init__.py", "config.py", "constants.py",
            "solver.py", "materials.py", "injection.py", "observables.py", "coupling.py",
            "custom_analysis.py", "geometry.py", "vertical.py", "three_d_cwt.py",
            "band_structure.py", "plotting.py", "io.py", "numerical_quality.py",
        }
        manifest["semiconductor_source_changed_during_run"] = [path for path in
            manifest["source_changed_during_run"] if Path(path).name in semiconductor_imported_names]
        manifest["hash_scope_note"] = "All src files were hashed, including unused Yb and triangular modules; the semiconductor import closure is checked separately."
        manifest["postprocessor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["environment"] = report["environment"]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

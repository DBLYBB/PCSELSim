"""Size scaling and dense-angle beam audit of the two saved triangular finalists.

The geometry and period remain fixed during the size scan. This tests a design
tradeoff, not size-independent single-mode lasing or global optimality. All
finite poles are passive; material intensity loss enters the gain audit once.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks, savgol_filter
from pcselsim.design_objectives import beam_metrics
from pcselsim.finite_discretization import staggered_grid_diagnostics
from pcselsim.run_output import prepare_output, write_provenance
from scripts.optimization.triangular_semiconductor_search import (
    Motif, evaluate_finite, json_write, write_csv, threadpool_limits)


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(input_dir):
    path=input_dir/"candidates.json"
    if not path.exists():
        path=input_dir/"summary.json"
    data=json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data,list) else [data],path


def dense_far_field(data, wavelength_nm, samples=301, view=1.):
    """Exactly the saved discrete radiation sums, with no eigenfield alteration."""
    angle = np.linspace(-view, view, samples)
    tx, ty = np.meshgrid(np.deg2rad(angle), np.deg2rad(angle))
    kx = (2*np.pi/(wavelength_nm*1e-9)*np.tan(tx)).ravel()
    ky = (2*np.pi/(wavelength_nm*1e-9)*np.tan(ty)).ravel()
    x, y = data["x_um"]*1e-6, data["y_um"]*1e-6
    rx, ry = data["all_radiation_x"][0], data["all_radiation_y"][0]
    fx, fy = np.empty(kx.size, complex), np.empty(kx.size, complex)
    for start in range(0, kx.size, 512):
        end = min(start+512, kx.size)
        phase = np.exp(-1j*(kx[start:end, None]*x+ky[start:end, None]*y))
        fx[start:end], fy[start:end] = phase@rx, phase@ry
    obliquity = np.cos(tx)+np.cos(ty)-1
    return angle, fx.reshape(tx.shape)*obliquity, fy.reshape(tx.shape)*obliquity


def extra_beam_metrics(angle, fx, fy):
    power = abs(fx)**2+abs(fy)**2
    power /= power.max()
    rr = np.linspace(0., float(angle[-1]), 1001)
    phi = np.linspace(0., 2*np.pi, 360, endpoint=False)
    tx, ty = rr[:, None]*np.cos(phi), rr[:, None]*np.sin(phi)
    step = float(angle[1]-angle[0])
    traces = map_coordinates(power, [(ty-angle[0])/step, (tx-angle[0])/step], order=1)
    crossings = []
    for trace in traces.T:
        below = np.flatnonzero(trace <= .5)
        if below.size and below[0] > 0:
            index = below[0]
            crossings.append(float(np.interp(.5, trace[index-1:index+1][::-1], rr[index-1:index+1][::-1])))
    radial = traces.mean(axis=1)
    smoothed = savgol_filter(radial, 15, 3)
    minima, _ = find_peaks(-smoothed, prominence=1e-5)
    minima = [int(i) for i in minima if rr[i] > max(crossings, default=.1) and smoothed[i] < .2]
    boundary = float(rr[minima[0]]) if minima else None
    xx, yy = np.meshgrid(angle, angle)
    radius = np.hypot(xx, yy)
    outside = radius >= boundary if boundary is not None else np.zeros(power.shape, bool)
    peak = float(power[outside].max()) if outside.any() else None
    pol = np.asarray([[np.sum(abs(fx)**2), np.sum(fx*np.conj(fy))],
                      [np.sum(fy*np.conj(fx)), np.sum(abs(fy)**2)]])
    eigen = np.linalg.eigvalsh(pol).real
    linear_x = float(np.sum(abs(fx)**2)/np.trace(pol).real)
    central = len(angle)//2
    cex, cey = fx[central, central], fy[central, central]
    central_total = float(abs(cex)**2+abs(cey)**2)
    stokes_v = float(-2*np.imag(cex*np.conj(cey))/central_total)
    return {"FWHM_diameter_min_deg": 2*min(crossings),
            "FWHM_diameter_max_deg": 2*max(crossings),
            "FWHM_diameter_ratio": max(crossings)/min(crossings),
            "FWHM_directions_tested": len(crossings),
            "first_radial_mean_minimum_deg": boundary,
            "outside_first_radial_mean_minimum_peak": peak,
            "outside_first_radial_mean_minimum_peak_dB": 10*np.log10(max(peak,1e-30)) if peak is not None else None,
            "outside_first_radial_mean_minimum_energy": float(power[outside].sum()/power.sum()) if outside.any() else None,
            "integrated_x_polarized_power_fraction": linear_x,
            "integrated_Jones_coherency_eigenvalue_contrast": float((eigen[1]-eigen[0])/eigen.sum()),
            "on_axis_normalized_Stokes_V": stokes_v,
            "definition": "Intensity roundness, not circular polarization. All powers normalized within +/-1 degree square. The side-lobe proxy uses a circular boundary at the first radial-mean intensity minimum; finite-angle window only. FWHM interpolated from 360 radial cuts.",
            "radial_angle_deg": rr.tolist(), "radial_mean_intensity": radial.tolist()}


def beam_stage(output, input_dir):
    source, _ = source_rows(input_dir)
    rows = []
    for row in source:
        path = input_dir/row["name"]/"N35"/"complex_fields.npz"
        if not path.exists():
            path=input_dir/"N35"/"complex_fields.npz"
        data = np.load(path)
        angle, fx, fy = dense_far_field(data, row["frozen_CWT_pole_wavelength_nm"])
        metrics = beam_metrics(angle, fx, fy, requested_view_deg=1., source_nyquist_deg=row["source_sampling_view_deg"])
        extra = extra_beam_metrics(angle, fx, fy)
        _, original_index, dense_index = np.intersect1d(np.round(data["angle_deg"],10),
            np.round(angle,10), return_indices=True)
        native_x = data["far_field_x"][np.ix_(original_index, original_index)]
        native_y = data["far_field_y"][np.ix_(original_index, original_index)]
        dense_x, dense_y = fx[np.ix_(dense_index,dense_index)], fy[np.ix_(dense_index,dense_index)]
        scale=max(float(np.linalg.norm(native_x)+np.linalg.norm(native_y)),1e-30)
        field_error=float((np.linalg.norm(dense_x-native_x)+np.linalg.norm(dense_y-native_y))/scale)
        if field_error > 1e-10:
            raise ValueError("Dense radiation transform does not reproduce the saved complex far field")
        record = {"name": row["name"], "source": str(path.relative_to(ROOT)), "source_sha256": source_hash(path),
                  "samples": len(angle), "actual_envelope_N": 35, "beam": metrics, "extra": extra,
                  "shared_angle_points_per_axis": len(original_index),
                  "relative_complex_field_error_at_shared_angles": field_error,
                  "161_to_301_angular_sampling_change": {key:metrics[key]-row["beam"][key]
                      for key in ("ellipticity","center_to_peak","radial_anisotropy_l2","r90_deg","encircled_0p5deg")}}
        rows.append(record)
        np.savez_compressed(output/(row["name"]+"_dense_beam.npz"), angle_deg=angle, far_field_x=fx, far_field_y=fy)
        power = abs(fx)**2+abs(fy)**2
        power /= power.max()
        fig, axes = plt.subplots(1,3,figsize=(12.5,3.9), constrained_layout=True)
        axes[0].imshow(power, origin="lower", extent=(-1,1,-1,1), cmap="inferno", vmin=0, vmax=1)
        axes[0].set(xlim=(-.5,.5), ylim=(-.5,.5), xlabel="theta x (deg)", ylabel="theta y (deg)", title=row["name"]+": linear intensity")
        axes[1].imshow(10*np.log10(np.maximum(power, 1e-6)), origin="lower", extent=(-1,1,-1,1), cmap="inferno", vmin=-40, vmax=0)
        if extra["first_radial_mean_minimum_deg"] is not None:
            axes[1].add_patch(plt.Circle((0,0), extra["first_radial_mean_minimum_deg"], fill=False, edgecolor="cyan", lw=.8))
        axes[1].set(xlabel="theta x (deg)", ylabel="theta y (deg)", title="Intensity (dB); first radial-mean minimum")
        center=len(angle)//2
        axes[2].semilogy(angle, np.maximum(power[center],1e-6), label="x cut")
        axes[2].semilogy(angle, np.maximum(power[:,center],1e-6), label="y cut")
        axes[2].semilogy(extra["radial_angle_deg"], np.maximum(extra["radial_mean_intensity"],1e-6), "--", label="radial mean")
        axes[2].set(xlabel="Angle (deg)", ylabel="Normalized intensity", ylim=(1e-5,1.1), title=f"FWHM {extra['FWHM_diameter_min_deg']:.3f}-{extra['FWHM_diameter_max_deg']:.3f} deg")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=.2)
        fig.savefig(output/(row["name"]+"_beam_audit.png"),dpi=190)
        plt.close(fig)
        json_write(output/"beam_audit.json", rows)
        print("beam", row["name"], extra["FWHM_diameter_min_deg"], extra["FWHM_diameter_max_deg"], extra["outside_first_radial_mean_minimum_peak_dB"],flush=True)
    return rows


def size_stage(output, input_dir):
    source, _ = source_rows(input_dir)
    rows = []
    for parent in source:
        base = Motif(**parent["spec"])
        for factor in (2/3, 1., 4/3, 2.):
            spec = replace(base, radius_um=base.radius_um*factor,
                           name=base.name+f"_R{base.radius_um*factor:.3f}")
            for cells in (19,27):
                start=time.perf_counter()
                row=evaluate_finite(spec,cells,10,2.,24,output/spec.name/f"N{cells}")
                row.update(parent_name=base.name, size_factor=factor, elapsed_seconds=time.perf_counter()-start,
                           interpretation="fixed geometry and period; local targets at all Gamma edges; not exhaustive finite spectrum")
                rows.append(row)
                json_write(output/"size_candidates.json",rows)
                print(spec.name,"N",cells,"g",round(row["g_th_cm"],4),"gap",round(row["g_th_gap_fraction"],4),"ellipse",round(row["ellipticity"],4),"eta",round(row["eta_rad"],4),flush=True)
    flat=[{k:r[k] for k in ("name","parent_name","size_factor","N","g_th_cm","g_th_gap_fraction","alpha_field_cm","eta_rad","ellipticity","center_to_peak","radial_anisotropy_l2","I_th_A","maximum_residual")}|{"radius_um":r["spec"]["radius_um"]} for r in rows]
    write_csv(output/"size_candidates.csv",flat)
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for parent in source:
        for cells,style in ((19,"o--"),(27,"s-")):
            series=[r for r in rows if r["parent_name"]==parent["name"] and r["N"]==cells]
            x=[r["spec"]["radius_um"] for r in series]
            for ax,key,label in zip(axes.ravel(),("g_th_cm","g_th_gap_fraction","eta_rad","ellipticity","radial_anisotropy_l2","I_th_A"),("Material gain threshold (cm$^{-1}$)","Relative threshold separation","Vertical radiation fraction","Second-moment ellipticity","Radial noncircularity L2","Conditional threshold current (A)")):
                ax.plot(x,[r[key] for r in series],style,label=parent["name"]+f" N{cells}")
                ax.set(xlabel="Aperture half-size (um)",ylabel=label);ax.grid(alpha=.2)
    axes[0,1].axhline(.1,color="grey",ls=":");axes[1,0].axhline(1.1,color="grey",ls=":")
    axes[0,0].legend(fontsize=7)
    fig.savefig(output/"size_tradeoffs.png",dpi=190);plt.close(fig)
    return rows


def refine_stage(output,input_dir):
    source,_=source_rows(input_dir)
    if len(source)!=1 or source[0]["name"]!="local_0067_950nm_400um":
        raise ValueError("refine stage requires the scaled_400um_finalist_v1 summary as --input")
    rows=[]
    for parent in source:
        spec=Motif(**parent["spec"])
        for order,dz in ((8,3.),(10,3.),(10,2.)):
            row=evaluate_finite(spec,19,order,dz,24,output/f"D{order}_dz{dz:g}_N19")
            rows.append(row);json_write(output/"refinements.json",rows)
            print("refine",order,dz,row["g_th_cm"],row["field_loss_gap_cm"],flush=True)
        scaled_path=ROOT/"results/current/optimization_20261021/triangular/size_scaling_v1/size_candidates.json"
        scaled=json.loads(scaled_path.read_text())
        n27=next(r for r in scaled if r["parent_name"]=="local_0067_950nm" and r["spec"]["radius_um"]==200. and r["N"]==27)
        combined=[]
        for rank in (0,1):
            values=np.array([r["spectrum"][rank]["alpha_per_m"]/100 for r in (rows[-1],n27,parent)])
            _,sensitivity,status=staggered_grid_diagnostics(1/np.array((19,27,35)),values)
            trunc=abs(rows[1]["spectrum"][rank]["alpha_per_m"]-rows[0]["spectrum"][rank]["alpha_per_m"])/100
            vertical=abs(rows[2]["spectrum"][rank]["alpha_per_m"]-rows[1]["spectrum"][rank]["alpha_per_m"])/100
            combined.append({"rank":rank,"lateral_grid_cm":sensitivity,"D8_to_D10_cm":trunc,"dz3_to_dz2_cm":vertical,"combined_cm":sensitivity+trunc+vertical,"mesh_status":status})
        json_write(output/"numerical_sensitivity.json",{"name":spec.name,"lowest_two":combined,
            "combined_lowest_two_cm":sum(r["combined_cm"] for r in combined),
            "N35_field_loss_gap_cm":parent["field_loss_gap_cm"],
            "ordering_resolved_on_tested_discretizations":parent["field_loss_gap_cm"]>sum(r["combined_cm"] for r in combined),
            "interpretation":"Empirical D, dz, lateral sensitivity, not a rigorous error bound or physical-model certification.",
            "size_source":str(scaled_path.relative_to(ROOT)),"size_source_sha256":source_hash(scaled_path)})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("beam","size","refine"),required=True)
    parser.add_argument("--input",type=Path,default=ROOT/"results/current/optimization_20261021/triangular/highmesh_retuned_v1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    args.input=args.input.resolve()
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    (output/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    json_write(output/"run_status.json",{"status":"running","stage":args.stage})
    _,input_path=source_rows(args.input)
    json_write(output/"run_plan.json",{"stage":args.stage,"input":str(input_path),"input_sha256":source_hash(input_path),"threads":1,"aperture_factors":[2/3,1.,4/3,2.],"finite_grids":[19,27],"D":10,"dz_nm":2.,"spectrum_scope":"local targets at all six Gamma edges; not exhaustive fine-grid spectrum"})
    try:
        with threadpool_limits(limits=1):
            rows={"beam":beam_stage,"size":size_stage,"refine":refine_stage}[args.stage](output,args.input)
    except Exception as error:
        json_write(output/"run_status.json",{"status":"failed","stage":args.stage,"exception":repr(error)})
        raise
    json_write(output/"run_status.json",{"status":"completed","stage":args.stage,"cases":len(rows)})


if __name__=="__main__":main()

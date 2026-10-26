"""Read saved final square fields and independently audit numerical evidence.

No optical coupling or eigenvalue search is repeated. The original sparse
pencil is rebuilt from saved C; vertical overlap is independently re-solved
only for the first two competitors. Coarse QR evidence is checked likewise.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.sparse.linalg import splu
from scipy.constants import e
from pcselsim.custom_analysis import finite_area_eigenproblem
from pcselsim.fabrication import evaluate_fabrication,inclusion_boundary
from pcselsim.mode_identity import independent_mode_indices
from pcselsim.polarization import polarization_metrics,stokes_maps
from pcselsim.frequency_cwt import background_detuning
from pcselsim.triangular_finite import TriangularFiniteSpec,triangular_finite_eigenproblem
from scripts.optimization.matched300_square_search import Design,build_cell,RULES
from scripts.optimization.square_semiconductor_search import BASE,build_stack,direct_vector_far_field

RESULT=ROOT/"results/current/matched300_20261021"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def numeric_differences(a,b):
    differences=[]
    if isinstance(a,dict):
        for key in a:
            if key in b:differences.extend(numeric_differences(a[key],b[key]))
    elif isinstance(a,(list,tuple)):
        for x,y in zip(a,b):differences.extend(numeric_differences(x,y))
    elif isinstance(a,(float,int,np.number)) and not isinstance(a,(bool,np.bool_)):
        differences.append(abs(float(a)-float(b)))
    elif isinstance(a,(str,bool)) or a is None:
        if a!=b:raise AssertionError("Non-numeric readback differs")
    return differences


def original_pencil(problem,fields,values):
    phi=np.asarray(fields).reshape(len(values),-1).T
    links=splu(problem.mass).solve(phi);mass_fields=problem.mass@links
    k_fields=problem.stiffness@links
    residual=np.linalg.norm(k_fields-mass_fields*np.asarray(values)[None,:],axis=0)/np.maximum(
        np.linalg.norm(k_fields,axis=0),abs(np.asarray(values))*np.linalg.norm(mass_fields,axis=0))
    norm=np.sum(abs(phi)**2,axis=0)
    local=(problem.local-problem.local.conj().T)/(2j)
    radiation=np.sum(phi.conj()*(local@phi),axis=0).real/norm
    transport=np.sum(phi.conj()*(problem.transport@links),axis=0).real/norm
    closure=abs(np.imag(values)-radiation-transport)
    return residual,radiation,transport,closure


def threshold(alpha_cm,gamma,neff,side=300.):
    gain=(2*alpha_cm+5.)/(gamma*BASE.ACTIVE_INDEX/neff)
    carrier=BASE.TRANSPARENCY_DENSITY_CM3*(BASE.MAXIMUM_GAIN_CM+
        gain*BASE.MAXIMUM_GAIN_CM/(-BASE.ZERO_CARRIER_GAIN_CM))/(BASE.MAXIMUM_GAIN_CM-gain)
    current=e*(BASE.ACTIVE_THICKNESS_NM*1e-7)*carrier/(BASE.CARRIER_LIFETIME_NS*1e-9)*(side*1e-4)**2
    return float(gain),float(current)


def audit_square():
    canonical=read(RESULT/"square/final_summary.json");results=[]
    paths=[]
    for key in ("portable_config","convergence_path","readback_path","source_provenance_path","source_independence_path"):
        paths.append(canonical[key])
    for candidate in canonical["candidates"]:
        summary=read(candidate["summary_path"]);geometry=read(candidate["actual_geometry_nm_path"])
        design=Design(**summary["design"]);cell=build_cell(design,96);stack=build_stack(design,cell)
        assert design.length_um==canonical["side_um"]==geometry["device_square_side_um"]==300.
        assert geometry["internal_intensity_loss_cm1"]==BASE.INTERNAL_LOSS_CM==5.
        assert (design.pc_nm,design.gaas_nm,design.qw_spacer_nm)==(190.,110.,25.)
        assert numeric_differences(geometry["stack"],asdict(stack))==[0.]*len(numeric_differences(geometry["stack"],asdict(stack)))
        fabrication=evaluate_fabrication(cell.inclusions,design.lattice_nm,rules=RULES)
        fabricated_difference=max(numeric_differences(fabrication,candidate["fabrication"]))
        assert fabricated_difference<1e-10 and fabrication["margin_pass"]
        geometry_differences=[]
        for inclusion,hole in zip(cell.inclusions,geometry["holes"],strict=True):
            actual=inclusion_boundary(inclusion,np.eye(2))*design.lattice_nm
            saved=np.asarray(hole.get("boundary_nm",hole.get("boundary_cartesian_nm")))
            geometry_differences.append(float(np.max(abs(actual-saved))))
        assert max(geometry_differences)<1e-10
        assert abs(summary["lowest"]["wavelength_nm"]-950.)<.05
        mode_rows=summary["frequency_closed_modes"];mode_checks=[];identity_fields=[];identity_values=[]
        source=summary["source_lateral_survey"]
        with np.load(candidate["final_field_path"],allow_pickle=False) as saved:
            assert all(np.isfinite(saved[key]).all() for key in saved.files)
            axis=saved["axis_deg"]
            for index,row in enumerate(mode_rows):
                name=row["mode"];field=saved[name+"_field"].copy()
                value=complex(row["delta_per_m"],row["alpha_field_cm1"]*100.)
                identity_fields.append(field);identity_values.append(value)
                check=dict(mode=name,wavelength_nm=row["wavelength_nm"],finite_field=True)
                gain,current=threshold(row["alpha_field_cm1"],row["Gamma"],row["effective_index"])
                check["gain_absolute_error_cm1"]=abs(gain-row["g_th_cm1"])
                check["current_absolute_error_A"]=abs(current-row["I_th_A"])
                frequency=background_detuning(row["wavelength_nm"],row["effective_index"],design.lattice_nm)-row["delta_per_m"]
                check["frequency_residual_per_m"]=float(frequency)
                assert abs(frequency)<.05 and check["gain_absolute_error_cm1"]<1e-10 and check["current_absolute_error_A"]<1e-12
                if index<2:
                    vertical=stack.solve_te0(row["wavelength_nm"],dz_nm=summary["dz_nm"])
                    check["independent_vertical_neff_error"]=abs(vertical.effective_index-row["effective_index"])
                    check["independent_vertical_Gamma_error"]=abs(vertical.confinement[BASE.ACTIVE_LAYER_NAME]-row["Gamma"])
                    assert max(check["independent_vertical_neff_error"],check["independent_vertical_Gamma_error"])<1e-11
                if name+"_coupling_m" in saved:
                    problem=finite_area_eigenproblem(SimpleNamespace(grid_points=summary["grid"],domain_um=300.,discretization="staggered"),saved[name+"_coupling_m"])
                    residual,local,transport,closure=original_pencil(problem,[field],[value])
                    check.update(original_pencil_relative_residual=float(residual[0]),loss_closure_per_m=float(closure[0]),
                        local_radiation_amplitude_loss_per_m=float(local[0]),lateral_amplitude_loss_per_m=float(transport[0]),
                        radiative_fraction_error=abs(2*local[0]/(2*value.imag+500)-row["radiative_extraction_fraction"]))
                    assert residual[0]<1e-10 and closure[0]<1e-7 and check["radiative_fraction_error"]<1e-10
                    fx,fy=saved[name+"_far_x"],saved[name+"_far_y"]
                    mode=SimpleNamespace(fields=field,radiation_x=saved[name+"_radiation_x"],radiation_y=saved[name+"_radiation_y"])
                    _,rebuilt_x,rebuilt_y,beam=direct_vector_far_field(mode,row["wavelength_nm"],300.,len(axis))
                    check["near_to_far_maximum_absolute_error"]=max(float(np.max(abs(fx-rebuilt_x))),float(np.max(abs(fy-rebuilt_y))))
                    check["beam_readback_maximum_numeric_error"]=max(numeric_differences(beam,row["beam"]))
                    check["polarization_readback_maximum_numeric_error"]=max(numeric_differences(polarization_metrics(axis,fx,fy,core_radius_deg=.2),row["polarization"]))
                    # A separate local Stokes formula checks the signed convention.
                    midpoint=len(axis)//2;x,y=fx[midpoint,midpoint],fy[midpoint,midpoint]
                    stokes=np.array([abs(x)**2-abs(y)**2,2*(x*y.conjugate()).real,2*(x*y.conjugate()).imag])/(abs(x)**2+abs(y)**2)
                    check["manual_on_axis_stokes_error"]=float(np.max(abs(stokes-row["polarization"]["on_axis"]["normalized_stokes_s1_s2_s3"])))
                    assert check["beam_readback_maximum_numeric_error"]<1e-12 and check["polarization_readback_maximum_numeric_error"]<1e-12
                mode_checks.append(check)
            # Distinct closed frequencies have different C(lambda), so check
            # any repeated numerical frequency cluster independently.
            independent=[]
            for index,row in enumerate(mode_rows):
                peers=[j for j in independent if abs(mode_rows[j]["wavelength_nm"]-row["wavelength_nm"])<=1e-7]
                selected=independent_mode_indices([identity_values[j] for j in peers]+[identity_values[index]],
                    [identity_fields[j] for j in peers]+[identity_fields[index]])
                assert len(selected)==len(peers)+1
                independent.append(index)
        paths.extend(candidate[key] for key in ("summary_path","actual_geometry_nm_path","final_field_path","overview_path","period_design_path","source_spectrum_path"))
        g1,g2=mode_rows[0]["g_th_cm1"],mode_rows[1]["g_th_cm1"]
        gap=g2/g1-1;igap=mode_rows[1]["I_th_A"]/mode_rows[0]["I_th_A"]-1
        assert abs(gap-candidate["relative_material_gap"])<1e-13 and abs(igap-candidate["relative_threshold_current_gap"])<1e-13
        results.append(dict(role=candidate["role"],recommended=candidate["recommended"],side_um=300.,
            g1_cm1=g1,g2_cm1=g2,material_gain_gap=gap,current_gap=igap,
            actual_geometry_maximum_absolute_error_nm=max(geometry_differences),fabrication_numeric_error_nm=fabricated_difference,
            modes_checked=mode_checks,independent_closed_modes=len(independent),
            source_census=dict(grid=source["grid"],distinct_poles=source["distinct_poles"],target_count=len(source.get("targets",[])),
                failures=len(source.get("failures",[])),scope=source["scope"]),
            unresolved_closure_count=candidate["unresolved_closure_count"],
            qualification="Verified first and nearest saved modes within retained lateral census; not an exhaustive fine-grid spectrum or nonlinear single-mode certificate."))
    files=[]
    for path in sorted(set(paths)):
        path=Path(path);assert path.is_file(),path
        files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    provenance=read(canonical["source_provenance_path"])
    sources={path:dict(recorded_sha256=digest,current_sha256=sha(path),matches=sha(path)==digest)
        for path,digest in provenance["code_sha256"].items()}
    assert all(item["matches"] for item in sources.values())
    return dict(canonical_summary_sha256=sha(RESULT/"square/final_summary.json"),all_checks_passed=True,
        candidates=results,linked_files=files,code_provenance=sources,
        current_vs_gain_gap_note="The same nominal carrier gain law produces a large transparency-current baseline: gain-gap percentages are not current-gap percentages.",
        no_new_eigensolve=True)


def audit_qr():
    results=[]
    for n in (5,7):
        comparison_path=RESULT/"triangular"/f"QR_validation_full_N{n}_comparison.json"
        comparison=read(comparison_path);checks=[]
        for backend in ("qr","qz"):
            folder=RESULT/"triangular"/f"QR_validation_full_N{n}_{backend}"
            path=folder/"qz_low_fields.npz"
            with np.load(path,allow_pickle=False) as saved:
                values=saved["eigenvalues"].copy();fields=saved["fields"].copy()
                problem,grid=triangular_finite_eigenproblem(SimpleNamespace(coupling_m=saved["coupling_m"]),
                    TriangularFiniteSpec(150.,n,0.,24,"square","staggered"))
                assert len(values)==len(fields)==problem.stiffness.shape[0]
                assert np.array_equal(saved["x_um"],grid.x_um) and np.array_equal(saved["y_um"],grid.y_um)
                residual,_,_,closure=original_pencil(problem,fields,values)
                rng=np.random.default_rng(1321);test=rng.normal(size=(len(values),3))+1j*rng.normal(size=(len(values),3))
                links=splu(problem.mass).solve(test)
                left=problem.stiffness@links;right=problem.local@test+1j*(problem.transport@links)
                identity=float(np.max(abs(left-right))/np.max(abs(left)))
                assert residual.max()<1e-10 and closure.max()<1e-6 and identity<1e-12
                checks.append(dict(backend=backend,matrix_dimension=len(values),retained_complete_modes=len(values),
                    independent_original_pencil_maximum_relative_residual=float(residual.max()),
                    independent_loss_budget_maximum_closure_per_m=float(closure.max()),
                    random_vector_equivalent_operator_identity_relative_error=identity,fields_sha256=sha(path)))
        assert comparison["one_to_one_assignment_maximum_relative_error"]<1e-10
        assert comparison["maximum_cluster_subspace_error"]<1e-6
        results.append(dict(N=n,comparison_sha256=sha(comparison_path),original_pencil_checks=checks,
            compared_complete_poles=comparison["full_pole_count"],
            assignment_relative_error=comparison["one_to_one_assignment_maximum_relative_error"],
            maximum_subspace_error=comparison["maximum_cluster_subspace_error"]))
    return dict(accepted_as_coarse_fine_guidance=True,checks=results,
        mathematical_basis="M invertible; H=K M^-1=C+i D M^-1; its right eigenvectors are coefficient-point phi=M u.",
        qualification="Private script only. N13 must retain dimension/finite/residual/negative-alpha counts and original-pencil checks; a full coarse spectrum remains guidance, not exhaustive fine spectrum.")


def main():
    output=RESULT/"independent_review";output.mkdir(parents=True,exist_ok=True)
    report=dict(script_sha256=sha(__file__),square=audit_square(),triangular_qr=audit_qr())
    target=output/"square_and_qr_evidence_audit.json"
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(path=str(target),square_candidates=3,qr_sizes=[5,7],all_checks_passed=True)))


if __name__=="__main__":main()

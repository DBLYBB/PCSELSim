"""Read saved final YAG fields back; check beams, pencils and flux.

中文：从正式复场回读本征残差、通量守恒与 Jones/Stokes；只对实际侧泵首模
重新构建物理耦合矩阵，其余模式使用各自保存的闭频矩阵检查原始本征方程。
"""
from matched300_yag_search import *
from types import SimpleNamespace
import hashlib


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    a=p.parse_args();summary=json.loads((a.case/"summary.json").read_text());rows=json.loads((a.case/"closed_optical_modes.json").read_text())
    lattice,design,period=summary["lattice"],summary["design"],summary["period_nm"]
    export_actual_geometry(lattice,design,period,summary["fabrication"],a.case/"actual_geometry_nm.json")
    settings=summary["settings"];grid=settings["square_grid"] if lattice=="square" else settings["tri_radius_cells"]
    data=np.load(a.case/"closed_fields.npz",allow_pickle=False)
    winner=summary["actual_winners"]["edge_dual"][0]
    # Rebuild at the shared-pump winner's actual wavelength, not a 1030-nm guess.
    # 使用实际赢家的闭合波长重建，避免把 Bragg 参考频率误作最终起振频率。
    opt,_,guide=optical(lattice,design,settings["order"],settings["dz_nm"],winner["wavelength_nm"],period,winner["closed_TE_family"])
    spec,_,_,mesh=finite(lattice,opt,grid,8,False)
    checks=[];far_checks=[]
    for row in rows:
        name=row["name"];field=data[name+"_field"];flat=field.reshape(len(opt.result.coupling_m),-1)
        norm=float(np.sum(abs(flat)**2));C=data[name+"_C"];Lup=data[name+"_Lup"];Ldown=data[name+"_Ldown"]
        if lattice=="square":problem=finite_area_eigenproblem(spec,C)
        else:problem,_=triangular_finite_eigenproblem(SimpleNamespace(coupling_m=C),spec)
        value=complex(row["delta_per_m"],row["alpha_field_per_m"])
        residual=relative_residual(problem.center_operator(),field,value)
        budget=problem.loss_budget(field)
        up,down=[float(2*np.einsum("ik,ij,jk->",flat.conj(),op,flat).real/norm) for op in (Lup,Ldown)]
        closure=float(np.linalg.norm((C-C.conj().T)/(2j)-Lup-Ldown)/max(np.linalg.norm(Lup+Ldown),1e-30))
        transport=2*budget["transport_amplitude_loss_per_m"]
        # Each loss term here is an intensity coefficient in inverse metres.
        # 此处通量、边界输运与 2*Im(eigenvalue) 均为功率损耗 m^-1。
        loss_res=abs(up+down+transport-2*value.imag)
        matrix_error=float(np.linalg.norm(C-opt.result.coupling_m)/max(np.linalg.norm(C),1e-30)) if name==winner["name"] else None
        record=dict(name=name,field_norm=norm,eigen_residual=residual,
            stored_frequency_residual_per_m=row["frequency_residual_per_m"],
            optical_flux_matrix_relative_residual=closure,discrete_boundary_flux_per_m=transport,
            passive_loss_closure_per_m=loss_res,up_loss_readback_per_m=up,down_loss_readback_per_m=down,
            gain_weight_readback=float(data[name+"_qgain"].sum()),
            main_fresh_coupling_relative_difference=matrix_error,
            minimum_side_operator_eigenvalue=float(min(np.linalg.eigvalsh(Lup).min(),np.linalg.eigvalsh(Ldown).min())))
        passed=(residual<2e-7 and abs(row["frequency_residual_per_m"])<.05 and closure<1e-8
            and loss_res<1e-4 and abs(up-row["up_power_loss_per_m"])<1e-7
            and abs(down-row["down_power_loss_per_m"])<1e-7 and transport>-1e-6
            and record["minimum_side_operator_eigenvalue"]>-1e-7
            and (matrix_error is None or matrix_error<1e-11))
        record["passed"]=bool(passed);checks.append(record)
        if name+"_far_x" in data:
            axis,fx,fy=data["angle_deg"],data[name+"_far_x"],data[name+"_far_y"]
            beam=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=row["beam"]["source_nyquist_deg"])
            pol=polarization_metrics(axis,fx,fy,core_radius_deg=.2)
            differences={k:abs(beam[k]-row["beam"][k]) for k in ("ellipticity","center_to_peak","radial_anisotropy_l2")}
            poldiff=max(abs(pol[region][key]-row["polarization"][region][key])
                for region in ("integrated","core") for key in ("degree_of_polarization","degree_of_linear_polarization","signed_circular_fraction_s3_over_s0"))
            far_checks.append(dict(name=name,beam=beam,polarization=pol,beam_absolute_differences=differences,
                polarization_max_absolute_difference=poldiff,
                passed=all(d<1e-8 for d in differences.values()) and poldiff<1e-8))
    cw=[]
    # This reads stationary budgets; it does not replay a time-domain trajectory.
    # 这里只回读静态能量和互补残差，不把它解释为时域稳定性验证。
    for pump in summary["pump_results"]:
        for result in pump["cw"]:
            if result.get("status","completed")!="completed":
                cw.append(dict(architecture=pump["architecture"],ratio=result["ratio"],passed=False,error=result.get("error")));continue
            budget=result["energy_budget_W"]
            passed=(abs(budget["energy_closure_residual_W"])<1e-8 and abs(budget["pump_flux_residual_W"])<1e-8
                    and result["ncp_residual"]<2e-7)
            cw.append(dict(architecture=pump["architecture"],ratio=result["ratio"],active_modes=result["active_modes"],
                energy_closure_residual_W=budget["energy_closure_residual_W"],pump_flux_residual_W=budget["pump_flux_residual_W"],
                ncp_residual=result["ncp_residual"],passed=bool(passed)))
    result=dict(status="passed" if all(r["passed"] for r in checks+far_checks+cw) else "failed",
        fields_sha256=sha(a.case/"closed_fields.npz"),geometry_sha256=sha(a.case/"actual_geometry_nm.json"),
        source_summary_sha256=sha(a.case/"summary.json"),modes=checks,actual_far_fields=far_checks,cw=cw,
        actual_edge_wavelength_target_pass=abs(winner["wavelength_nm"]-1030.)<=.05,
        nominal_aperture_um=[300,300],scope="Saved-state optical, boundary flux, Jones/Stokes and stationary energy readback; fresh physical coupling independently rebuilt for edge first mode")
    if not result["actual_edge_wavelength_target_pass"]:result["status"]="failed"
    save(a.case/"readback_qa.json",result);print(result["status"],len(checks),len(far_checks),len(cw),flush=True)
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()

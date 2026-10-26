"""Bounded two-REAL-mode YAG refinement with independent frequency closures.

Track both the known bright and known dark mode. A round bright branch alone
never establishes first lasing. This screen uses precisely32 motifs and only
clear improvements may enter a new complete QZ/stripe/32-mode final census.

中文：仅跟踪已知亮模与暗模的 32 点局部筛选；两模通过不代表完整谱通过，
入选结构仍必须重新进行更广泛的独立模式搜寻和频率闭合。
"""
from matched300_yag_search import *
from scipy.stats import qmc


PARAMETERS=("main_share","angle2_deg","distance1","rounding","main_orientation_deg")
SCALES=np.array([.03,1.,.01,.04,4.])


def proposals(d):
    # Equal distances and opposite angles preserve this two-satellite family.
    # 此局部族约束两副孔等距、角度互为正负；它不是所有三孔布局的全局搜索。
    yield "nominal",dict(d),np.zeros(5)
    for k,(key,scale) in enumerate(zip(PARAMETERS,SCALES)):
        for delta in (-scale,-scale/2,scale/2,scale):
            change=np.zeros(5);change[k]=delta
            design={**d,**{name:float(d[name]+step) for name,step in zip(PARAMETERS,change)}}
            design.update(angle1_deg=-design["angle2_deg"],distance2=design["distance1"])
            yield f"{key}_{delta:+g}",design,change
    for i,unit in enumerate(qmc.LatinHypercube(5,seed=3201030).random(11)):
        change=(2*unit-1)*SCALES
        design={**d,**{key:float(d[key]+step) for key,step in zip(PARAMETERS,change)}}
        design.update(angle1_deg=-design["angle2_deg"],distance2=design["distance1"])
        yield f"joint_{i:02d}",design,change


def geometric_seed(d,target,period,reference,source_spec,mesh,seed):
    """Adiabatic actual-geometry steps, keeping strict local field gates."""
    value,field=seed;model=reference;fraction=0.;step=1.;trace=[]
    wavelength=reference.result.bragg_wavelength_nm
    while fraction<1.-1e-12:
        proposal=min(1.,fraction+step)
        intermediate={**target,**{k:float(d[k]+proposal*(target[k]-d[k])) for k in PARAMETERS}}
        intermediate.update(angle1_deg=-intermediate["angle2_deg"],distance2=intermediate["distance1"])
        candidate_model,_,_=optical("triangular",intermediate,32,1.,wavelength,period)
        _,_,problem,_=finite("triangular",candidate_model,52,8,False)
        flat=field.reshape(6,-1);change=candidate_model.result.coupling_m-model.result.coupling_m
        predictor=value+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
        try:
            v,f,audit=follow_reference_pole(problem,field,predictor)
            if audit["interpolated_field_overlap"]<.75:raise RuntimeError("Geometry local overlap below0.75")
        except Exception:
            step/=2
            if step<1/64:raise RuntimeError("Geometry branch lost under bounded1/64 continuation; no relaxed gate")
            continue
        fraction=proposal;model,value,field=candidate_model,v,f
        trace.append(dict(fraction=fraction,**audit));step=min(2*step,1.-fraction)
    return model,(value,field),trace


def close_seed_newton(design,period,reference,seed,spec,mesh):
    """Exact-residual checked group-index Newton; original Brent is fallback.

    Each trial uses fresh physical C(lambda), adaptive local field continuation,
    and the unmodified finite pencil. No frequency or overlap gate is relaxed.

    中文：群折射率仅用于 Newton 步长预测；接受条件仍用实际 C(lambda) 重算
    的精确残差。预测失败回退原 Brent 流程，不放宽任何物理残差或分支重叠门限。
    """
    value0,field0=seed;ref=float(reference.result.bragg_wavelength_nm);trace=[];states={}
    guide=structure("triangular",design)[1]
    beta0=4*np.pi/(np.sqrt(3)*period*1e-9)
    def state(w,model,value,field,audit):
        beta=2*np.pi*model.result.effective_index/(w*1e-9)
        residual=float((beta**2-beta0**2)/(2*beta0)-value.real)
        problem,_=triangular_finite_eigenproblem(model.result,spec)
        er=relative_residual(problem.center_operator(),field,value)
        if er>2e-7 or value.imag< -1e-7:raise RuntimeError("Newton trial fails original passive pencil")
        mode=replace(as_mode("triangular",field,value,model.result,mesh,"closed"),eigen_residual=er)
        result=dict(wavelength=float(w),model=model,value=value,field=field,mode=mode,residual=residual)
        states[float(w)]=result
        trace.append(dict(wavelength_nm=float(w),frequency_residual_per_m=residual,
            delta_per_m=float(value.real),alpha_field_per_m=float(value.imag),eigen_residual=er,
            overlap=float(audit.get("interpolated_field_overlap",1.)),tracking_solver=audit.get("solver","validated_reference_pole"),
            frequency_root_solver="exact_C_lambda_group_index_Newton_with_original_Brent_fallback",
            optical_theorem_residual=model.optical_theorem_residual))
        return result
    state(ref,reference,value0,field0,{})
    def evaluate(w):
        w=float(w)
        if w in states:return states[w]
        previous=min(states.values(),key=lambda s:abs(s["wavelength"]-w));steps=0
        while w not in states:
            delta=w-previous["wavelength"];step=np.sign(delta)*min(abs(delta),1.)
            while True:
                trial=w if abs(step)>=abs(delta)*(1-1e-12) else previous["wavelength"]+step
                try:
                    model,_,_=optical("triangular",design,32,1.,trial,period)
                    problem,_=triangular_finite_eigenproblem(model.result,spec)
                    flat=previous["field"].reshape(6,-1)
                    prediction=previous["value"]+np.einsum("ik,ij,jk->",flat.conj(),model.result.coupling_m-previous["model"].result.coupling_m,flat)/np.sum(abs(flat)**2)
                    value,field,audit=follow_reference_pole(problem,previous["field"],prediction)
                    if audit["interpolated_field_overlap"]<.75:raise RuntimeError("Newton local frequency overlap below0.75")
                    previous=state(trial,model,value,field,audit);break
                except Exception:
                    step/=2
                    if abs(step)<2e-4:raise
            steps+=1
            if steps>3000:raise RuntimeError("Newton bounded frequency continuation exceeded")
        return states[w]
    try:
        current=states[ref]
        for iteration in range(7):
            if abs(current["residual"])<=.005:
                model=current["model"];wave=current["wavelength"]
                trace.append({**trace[-1],"closure_complete":True})
                return wave,model,guide,spec,current["mode"],trace
            wave=current["wavelength"];model=current["model"]
            beta=2*np.pi*model.result.effective_index/(wave*1e-9)
            derivative=-(beta/beta0)*2*np.pi*model.result.group_index*1e9/wave**2
            increment=-current["residual"]/derivative
            if abs(increment)>.5 or not 1000<=wave+increment<=1060:break
            current=evaluate(wave+increment)
    except Exception as error:trace.append(dict(newton_attempt_error=repr(error)))
    answer=close_seed("triangular",design,period,reference,seed,spec,mesh,32,1.)
    return (*answer[:-1],trace+answer[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source_case",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fast-frequency",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--validate-only",action="store_true");a=p.parse_args()
    summary=json.loads((a.source_case/"summary.json").read_text());source_rows=json.loads((a.source_case/"closed_modes.json").read_text())[:2]
    if len(source_rows)!=2:raise ValueError("Exactly the two independently validated source fields are required")
    d=summary["design"];period=summary["fixed_actual_period_nm"];saved=np.load(a.source_case/"closed_fields.npz")
    references=[]
    for row in source_rows:
        model,_,_=optical("triangular",d,32,1.,row["wavelength_nm"],period)
        spec,_,_,mesh=finite("triangular",model,52,8,False)
        references.append((row,model,spec,mesh,saved[row["name"]+"_field"].copy()))
    plans=list(proposals(d));assert len(plans)==32
    if a.validate_only:
        a.output.mkdir(parents=True,exist_ok=True);validation=[]
        for index in (0,1):
            _,target,_=plans[index]
            for source,reference,spec,mesh,field in references:
                seed=(complex(source["delta_per_m"],source["alpha_field_per_m"]),field)
                model,continued,_=geometric_seed(d,target,period,reference,spec,mesh,seed)
                t=time.perf_counter();old=close_seed("triangular",target,period,model,continued,spec,mesh,32,1.);old_s=time.perf_counter()-t
                t=time.perf_counter();new=close_seed_newton(target,period,model,continued,spec,mesh);new_s=time.perf_counter()-t
                old_gain=(2*old[4].alpha_per_m+10.)/active_weights(old[1].result,structure("triangular",target)[1],24)[2].sum()/100
                new_gain=(2*new[4].alpha_per_m+10.)/active_weights(new[1].result,structure("triangular",target)[1],24)[2].sum()/100
                f,g=old[4].fields.ravel(),new[4].fields.ravel();g=g*np.exp(1j*np.angle(np.vdot(g,f)))
                field_error=float(np.linalg.norm(f-g)/np.linalg.norm(f));dw=abs(old[0]-new[0]);dg=abs(old_gain-new_gain)/old_gain
                row=dict(case_index=index,source_mode=source["name"],old_seconds=old_s,new_seconds=new_s,
                    old_wavelength_nm=old[0],new_wavelength_nm=new[0],wavelength_abs_difference_nm=dw,
                    old_g_cm1=old_gain,new_g_cm1=new_gain,gain_relative_difference=dg,phase_aligned_field_relative_error=field_error,
                    old_frequency_residual_per_m=old[-1][-1]["frequency_residual_per_m"],new_frequency_residual_per_m=new[-1][-1]["frequency_residual_per_m"],
                    new_eigen_residual=new[4].eigen_residual,
                    passed=dw<5e-7 and dg<1e-7 and field_error<1e-6 and abs(new[-1][-1]["frequency_residual_per_m"])<=.005)
                validation.append(row);save(a.output/"validation.json",dict(status="passed" if all(r["passed"] for r in validation) else "failed",results=validation))
                print("validation",index,source["name"],row,flush=True)
        if not all(r["passed"] for r in validation):raise SystemExit(1)
        return
    a.output.mkdir(parents=True,exist_ok=a.resume)
    plan=dict(source=str(a.source_case),count=32,fixed_period_nm=period,
        parameters=list(PARAMETERS),scales=SCALES,nominal_design=d,
        goal="Both actual closed branches; bright first, axial<=1.05,center>=0.98,radial<=0.10,g<=9.5,prefer cold gap>=10%; never replace complete formal census")
    results=json.loads((a.output/"all_results.json").read_text()) if a.resume and (a.output/"all_results.json").exists() else []
    if a.resume:
        original=json.loads((a.output/"plan.json").read_text())
        if original["nominal_design"]!=d or original["source"]!=str(a.source_case):raise ValueError("Resume source mismatch")
        plan["previous_completed_evaluations"]=len(results)
    plan["frequency_solver"]="exact_residual_group_index_Newton_with_Brent_fallback" if a.fast_frequency else "adaptive_Brent"
    save(a.output/"plan.json",plan)
    best_score=min((r["score"] for r in results if r.get("score") is not None),default=np.inf)
    for index,(name,design,changes) in enumerate(plans):
        if any(r["index"]==index for r in results):continue
        start=time.monotonic();fab=evaluate_fabrication(geometry("triangular",design).inclusions,period,TRIANGULAR_DIRECT,RULE)
        result=dict(index=index,name=name,lattice="triangular",design=design,period_nm=period,changes=changes,
            settings=dict(order=32,dz_nm=1.,grid=52),fabrication=fab)
        models=[];rows=[];names=[];arrays={};stored=[];failures=[]
        if not fab["margin_pass"]:
            result.update(status="geometry_rejected",seconds=time.monotonic()-start)
        else:
            for source,reference,spec,mesh,field in references:
                try:
                    seed=(complex(source["delta_per_m"],source["alpha_field_per_m"]),field)
                    model,seed,geometry_trace=geometric_seed(d,design,period,reference,spec,mesh,seed)
                    if a.fast_frequency:
                        w,opt,g,cs,mode,trace=close_seed_newton(design,period,model,seed,spec,mesh)
                    else:w,opt,g,cs,mode,trace=close_seed("triangular",design,period,model,seed,spec,mesh,32,1.)
                    mode=replace(mode,name=source["name"])
                    row,data=mode_row("triangular",opt,g,cs,mode,w,201);axis,fx,fy,widths,qe,qg=data
                    up,down=opt.side_power_losses(mode.fields)
                    lateral=binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31)
                    models.append(SpatialYbYAGModel(material(w),widths,qe,2*mode.alpha_per_m+10.,up,down,c/opt.result.group_index,qg,lateral));names.append(mode.name)
                    row.update(wavelength_nm=w,closed_TE_family=0,frequency_residual_per_m=trace[-1]["frequency_residual_per_m"],
                        geometry_trace=geometry_trace,frequency_trace=trace,source_mode=source["name"])
                    rows.append(row);stored.append((mode,axis,fx,fy))
                    arrays.update({mode.name+"_field":mode.fields,mode.name+"_far_x":fx,mode.name+"_far_y":fy,
                        mode.name+"_C":opt.result.coupling_m,mode.name+"_qgain":qg,mode.name+"_qenergy":qe,
                        mode.name+"_lateral":lateral,mode.name+"_widths":widths,
                        mode.name+"_Lup":opt.upward_amplitude_loss_m,mode.name+"_Ldown":opt.downward_amplitude_loss_m})
                except Exception as error:failures.append(dict(source_mode=source["name"],error=repr(error)))
            if len(rows)==2:
                retained,identity=distinct_closed_modes(rows,stored)
                if len(retained)!=2:result.update(status="branches_merged",mode_identity_audit=identity,closed_modes=rows)
                else:
                    system=SpatialMultimodeYbYAG(tuple(models),tuple(names));pumps=[]
                    for architecture in ("edge_dual","top_single"):
                        threshold=shared_inversion_pump_threshold(system,PumpGeometryConfig(architecture,0.,300.,31))
                        pumps.append(dict(architecture=architecture,threshold=asdict(threshold)))
                    first_names=pumps[0]["threshold"]["first_modes"]
                    if len(first_names)!=1:result.update(status="ambiguous_or_unreachable_first",pumps=pumps,closed_modes=rows)
                    else:
                        first=next(r for r in rows if r["name"]==first_names[0]);other=next(r for r in rows if r["name"]!=first_names[0])
                        # Evaluate the actual edge-pump winner's beam, even if it is dark.
                        # 光斑指标取共享侧泵实际首模；不能先指定亮模再忽略阈值更低的暗模。
                        beam=first["beam"];gain=first["required_material_gain_cm1"]
                        gap=other["required_material_gain_cm1"]/gain-1
                        bright=beam["ellipticity"]<=1.05 and beam["center_to_peak"]>=.98 and beam["radial_anisotropy_l2"]<=.10
                        eligible=bright and gain<=9.5 and gap>=.10
                        penalty=20*max(beam["ellipticity"]-1.05,0)+30*max(.98-beam["center_to_peak"],0)+20*max(beam["radial_anisotropy_l2"]-.1,0)+4*max(.1-gap,0)+max(gain-9.5,0)
                        score=gain*(1+penalty)
                        result.update(status="actual_closed_two_mode_screen",lowest=first,wavelength_nm=first["wavelength_nm"],closed_modes=rows,
                            pumps=pumps,cold_gap_relative_to_edge_winner=gap,bright_first=bool(bright),goal_met_in_two_mode_set=bool(eligible),
                            score=score,mode_identity_audit=identity,
                            scope="Both supplied physical branches independently closed; shared pump first among these two only, not a complete spectrum")
                        if score<best_score:
                            best_score=score;arrays.update(angle_deg=axis,x_um=stored[0][0].grid.x_um,y_um=stored[0][0].grid.y_um)
                            np.savez_compressed(a.output/"best_so_far_fields.npz",**arrays);save(a.output/"best_so_far.json",result)
            else:result.update(status="branch_failed",closed_modes=rows,failures=failures)
            result["seconds"]=time.monotonic()-start
        results.append(result);save(a.output/"all_results.json",results)
        save(a.output/"ranked.json",sorted((r for r in results if r.get("lowest")),key=lambda r:(not r["goal_met_in_two_mode_set"],r["score"])))
        print(index,name,result["status"],result.get("goal_met_in_two_mode_set"),result.get("cold_gap_relative_to_edge_winner"),
            result.get("lowest",{}).get("required_material_gain_cm1"),result["seconds"],flush=True)
    save(a.output/"summary.json",dict(status="bounded_screen_complete",count=len(results),
        goal_met_cases=[r["index"] for r in results if r.get("goal_met_in_two_mode_set")],
        source=str(a.source_case),scope="32 motifs maximum. Only a clear two-mode improvement may be promoted to a new complete77-target/32-independent-mode formal validation."))


if __name__=="__main__":main()

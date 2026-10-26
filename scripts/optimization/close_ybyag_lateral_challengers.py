"""Frequency-close saved full-spectrum challengers and share their inversion."""
import sys,json,argparse,shutil,time
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from optimize_ybyag_cavity import Design,structure,save,spatial_model,beam
from evaluate_ybyag_saved_modes import load_closed
from pcselsim.custom_analysis import FourWaveOpticalSpec,finite_area_eigenproblem,LinearMode
from pcselsim.finite_discretization import relative_residual
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.frequency_cwt import background_detuning
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig,pump_geometry_metadata
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.run_output import prepare_output,write_provenance,default_output


def closed_duplicate_name(mode,wavelength,coupling,rows,folder,family=0):
    """Numerical identity only at the same frequency, TE family and operator.

    Nonnormal modes may be almost parallel without being the same complex
    pole. High field overlap alone never removes a challenger.
    中文：只对同频、同纵模族、同算符的复极点判断身份，不用高重叠删除竞争模。
    """
    value=complex(mode.delta_per_m,mode.alpha_per_m)
    for row in rows:
        if row.get('te_family',0)!=family or abs(row['wavelength_nm']-wavelength)>2e-7:continue
        with np.load(Path(folder)/(row['name']+'_fields.npz'),allow_pickle=False) as saved:
            if 'C' not in saved or saved['field'].shape!=mode.fields.shape:continue
            if not np.allclose(saved['C'],coupling,rtol=1e-10,atol=1e-8):continue
            other=saved['field'].copy()
        delta=row.get('delta_per_m')
        if delta is None:
            evaluations=row.get('frequency_evaluations',())
            if not evaluations:continue
            delta=evaluations[-1].get('finite_detuning_per_m')
        if delta is None:continue
        previous=complex(delta,row['amplitude_loss_per_m'])
        if not independent_eigenmode(value,mode.fields,[previous],[other]):return row['name']
    return None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('challenge');p.add_argument('--output')
    p.add_argument('--count',type=int,default=16);p.add_argument('--truncation',type=int,default=4)
    p.add_argument('--vertical-step-nm',type=float,default=5.)
    a=p.parse_args(argv);source=Path(a.source);challenge=Path(a.challenge)
    out=prepare_output(a.output or default_output(ROOT,'optimization_20261021/yag/challengers_closed'),ROOT)
    write_provenance(out,ROOT);save(out/'run_status.json',dict(status='running'))
    old=json.loads((source/'summary.json').read_text());design=Design(**old['design']);period=old['fixed_period_nm']
    diagnostic=json.loads((challenge/'spectrum_challenge.json').read_text());reference=diagnostic['reference_wavelength_nm']
    cell,guide,stack,_=structure(design);n=np.load(challenge/f"lateral_{diagnostic['candidates'][0]['spectrum_index']}_seed.npz")['field'].shape[-1]
    _,original_system=load_closed(source);models=list(original_system.modes);names=list(original_system.names)
    rows=json.loads((source/'optical_modes.json').read_text())
    for r in rows:shutil.copy2(source/(r['name']+'_fields.npz'),out/(r['name']+'_fields.npz'))
    cache={}
    def optical_at(wavelength):
        key=float(wavelength)
        if key not in cache:
            opt=build_reflector_coupling(cell,guide,stack,'passive YAG photonic crystal',period,reference,
                ThreeDCWTSettings(a.truncation,a.vertical_step_nm),fixed_wavelength_nm=key)
            r=opt.result;spec=FourWaveOpticalSpec(key,r.effective_index,r.group_index,design.device_um,.1,
                tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),n)
            cache[key]=opt,spec,finite_area_eigenproblem(spec,r.coupling_m)
        return cache[key]
    reference_optical,_,_=optical_at(reference)
    for candidate in diagnostic['candidates'][:a.count]:
        index=candidate['spectrum_index'];seedfile=challenge/f'lateral_{index}_seed.npz'
        seed=np.load(seedfile)['field'];seedvalue=complex(np.load(seedfile)['value'])
        # A high-overlap seed can be a distinct nonnormal pole. Close it first;
        # numerical identity is checked against the matching closed operator.
        trace=[];evals={}
        def evaluate(wavelength):
            key=float(wavelength)
            if key not in evals:
                opt,spec,problem=optical_at(key);change=opt.result.coupling_m-reference_optical.result.coupling_m
                flat=seed.reshape(4,-1);target=seedvalue+np.einsum('ik,ij,jk->',flat.conj(),change,flat)/np.sum(abs(flat)**2)
                for count,dense in ((6,400),(24,2500)):
                    values,fields=problem.near(target,count,seed.ravel(),tol=1e-10,maxiter=500,dense_limit=dense)
                    overlaps=abs(seed.ravel().conj()@fields)**2/np.sum(abs(seed)**2)
                    selected=int(np.argmax(overlaps))
                    if overlaps[selected]>.1:break
                if overlaps[selected]<=.1:raise RuntimeError(f'Lateral {index} lost reference overlap')
                field=fields[:,selected].reshape(seed.shape);value=values[selected];rx,ry=opt.result.radiation_fields(field)
                mode=LinearMode(f'TE0_L{index}',value.real,value.imag,field,0,rx,ry,discretization='staggered',
                    eigen_residual=relative_residual(problem.center_operator(),field,value))
                gap=background_detuning(key,opt.result.effective_index,period)-value.real
                evals[key]=opt,spec,mode,float(overlaps[selected]),float(gap)
            return evals[key]
        def gap(wavelength):
            opt,spec,mode,overlap,value=evaluate(wavelength)
            trace.append(dict(wavelength_nm=float(wavelength),group_index=spec.group_index,
                effective_index=spec.effective_index,
                background_detuning_per_m=background_detuning(wavelength,spec.effective_index,period),
                finite_detuning_per_m=mode.delta_per_m,residual_per_m=value,overlap=overlap,
                eigen_residual=mode.eigen_residual,optical_theorem_residual=opt.optical_theorem_residual))
            return value
        half=.5
        while gap(reference-half)*gap(reference+half)>0:
            half*=2
            if half>16:raise RuntimeError('No challenger frequency root within the bounded study window')
        wavelength=brentq(gap,reference-half,reference+half,xtol=2e-7,rtol=1e-12)
        opt,spec,mode,overlap,residual=evaluate(wavelength)
        if abs(residual)>.05:raise RuntimeError('Challenger frequency residual exceeds .05/m')
        duplicate=closed_duplicate_name(mode,wavelength,opt.result.coupling_m,rows,out)
        if duplicate:
            print('duplicate',index,duplicate,flush=True);continue
        model=spatial_model(design,opt,mode,wavelength);models.append(model);names.append(mode.name)
        row=dict(name=mode.name,te_family=0,branch=f'L{index}',wavelength_nm=wavelength,
            amplitude_loss_per_m=mode.alpha_per_m,total_power_loss_per_m=model.total_power_loss_per_m,
            up_loss_per_m=model.upward_power_loss_per_m,down_loss_per_m=model.downward_power_loss_per_m,
            gain_weight=float(model.optical_gain_weights.sum()),energy_overlap=float(model.optical_energy_fractions.sum()),
            frequency_residual_per_m=residual,eigen_residual=mode.eigen_residual,reference_overlap=overlap,
            ffp=beam(spec,mode),frequency_evaluations=trace)
        rows.append(row);save(out/'optical_modes.json',rows)
        np.savez_compressed(out/(mode.name+'_fields.npz'),field=mode.fields,radiation_x=mode.radiation_x,radiation_y=mode.radiation_y,
            qgain=model.optical_gain_weights,qenergy=model.optical_energy_fractions,widths=model.cell_widths_m,
            lateral=model.lateral_energy_fractions,C=opt.result.coupling_m,Lup=opt.upward_amplitude_loss_m,Ldown=opt.downward_amplitude_loss_m)
        print('closed',mode.name,wavelength,mode.alpha_per_m,'overlap',overlap,flush=True)
    system=SpatialMultimodeYbYAG(tuple(models),tuple(names));results=[]
    for arch,passes,eta in [('edge_dual',1,1.),('top_single',20,.98)]:
        pump=PumpGeometryConfig(arch,0,design.device_um,31,top_passes=passes,return_efficiency=eta)
        threshold=shared_inversion_pump_threshold(system,pump);scans=[]
        for multiplier in (.8,1.05,1.2,1.5,2.,4.):
            power=multiplier*threshold.first_threshold_internal_W
            fit=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
            scans.append(dict(threshold_multiplier=multiplier,input_W=power,active_modes=fit.active_modes,
                outputs_W=dict(zip(names,map(float,fit.upward_output_W))),budget=fit.energy_budget_W,
                ncp_residual=fit.complementarity_residual,population_residual_per_s=fit.population_residual_per_s,
                inactive_gain_excess_per_m=fit.maximum_inactive_gain_excess_per_m,
                pump_geometry=pump_geometry_metadata(models[0],replace(pump,internal_incident_power_W=power))))
        results.append(dict(architecture=arch,threshold=asdict(threshold),cw_scan=scans));save(out/'pump_results.json',results)
    save(out/'summary.json',dict(design=asdict(design),fixed_period_nm=period,included_modes=names,
        isolated_TE_count=old['isolated_TE_count'],included_TE_families=[0],pump_results=results,
        numerics=dict(grid_points=n,CWT_truncation=a.truncation,vertical_step_nm=a.vertical_step_nm,
                      retained_source_numerics=old.get('numerics')),
        scope='additional lateral seeds selected from full fixed-frequency finite spectrum, then individually frequency closed; not exhaustive global poles',
        source_retained=str(source),source_challenge=str(challenge)))
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()

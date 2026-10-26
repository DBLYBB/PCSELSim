"""Re-close saved lateral poles on a finer mesh and CWT truncation.

Interpolated fields are only tracking seeds. Every accepted result solves the
new K/M problem, passes its actual eigen-residual, and closes its frequency at
the same fixed physical period. This is continuation, not a new exhaustive
enumeration of the finer-grid spectrum.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from optimize_ybyag_cavity import Design,structure,save,spatial_model,beam
from close_ybyag_lateral_challengers import closed_duplicate_name
from pcselsim.custom_analysis import FourWaveOpticalSpec,finite_area_eigenproblem,LinearMode,plot_vector_far_field_diagnostics
from pcselsim.finite_discretization import relative_residual
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.frequency_cwt import background_detuning
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig,pump_geometry_metadata
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.run_output import prepare_output,write_provenance


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('--output',required=True)
    p.add_argument('--grid',type=int,default=37);p.add_argument('--truncation',type=int,default=10)
    p.add_argument('--vertical-step-nm',type=float,default=2.5)
    a=p.parse_args(argv);source=Path(a.source);old=json.loads((source/'summary.json').read_text())
    design=Design(**old['design']);period=old['fixed_period_nm']
    out=prepare_output(a.output,ROOT);write_provenance(out,ROOT)
    save(out/'run_status.json',dict(status='running'))
    cell,guide,stack,_=structure(design);settings=ThreeDCWTSettings(a.truncation,a.vertical_step_nm)
    rows=[];models=[];names=[];mapping=[];stored=[]
    for oldrow in json.loads((source/'optical_modes.json').read_text()):
        data=np.load(source/(oldrow['name']+'_fields.npz'))
        original=data['field'];ratio=a.grid/original.shape[-1]
        seed=zoom(original.real,(1,ratio,ratio),order=1)+1j*zoom(original.imag,(1,ratio,ratio),order=1)
        seed/=np.linalg.norm(seed)
        evaluation=min(oldrow['frequency_evaluations'],key=lambda r:abs(r['wavelength_nm']-oldrow['wavelength_nm']))
        seedvalue=complex(evaluation['finite_detuning_per_m'],oldrow['amplitude_loss_per_m'])
        cache={};trace=[]
        def evaluate(wavelength):
            key=float(wavelength)
            if key in cache:return cache[key]
            opt=build_reflector_coupling(cell,replace(guide,te_family=oldrow['te_family']),stack,
                'passive YAG photonic crystal',period,oldrow['wavelength_nm'],settings,fixed_wavelength_nm=key)
            r=opt.result
            spec=FourWaveOpticalSpec(key,r.effective_index,r.group_index,design.device_um,.1,
                tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),a.grid)
            problem=finite_area_eigenproblem(spec,r.coupling_m)
            flat=seed.reshape(4,-1);delta=r.coupling_m-data['C']
            target=seedvalue+np.einsum('ik,ij,jk->',flat.conj(),delta,flat)/np.sum(abs(flat)**2)
            best=0.
            for count in (8,24,48):
                values,fields=problem.near(target,count,seed.ravel(),tol=1e-10,maxiter=600,dense_limit=400)
                overlaps=abs(seed.ravel().conj()@fields)**2
                chosen=int(np.argmax(overlaps));best=float(overlaps[chosen])
                if best>.4:break
            if best<.2:raise RuntimeError(f"lost seed {oldrow['name']}: overlap {best}")
            field=fields[:,chosen].reshape(seed.shape);value=values[chosen]
            if value.imag<0:raise RuntimeError('negative passive loss')
            rx,ry=r.radiation_fields(field)
            mode=LinearMode(oldrow['name'],float(value.real),float(value.imag),field,best,rx,ry,
                discretization='staggered',eigen_residual=relative_residual(problem.center_operator(),field,value))
            residual=background_detuning(key,r.effective_index,period)-value.real
            cache[key]=opt,spec,mode,best,float(residual)
            return cache[key]
        def gap(wavelength):
            opt,spec,mode,overlap,residual=evaluate(wavelength)
            trace.append(dict(wavelength_nm=float(wavelength),group_index=spec.group_index,
                effective_index=spec.effective_index,finite_detuning_per_m=mode.delta_per_m,
                residual_per_m=residual,field_overlap=overlap,eigen_residual=mode.eigen_residual))
            return residual
        reference=oldrow['wavelength_nm'];half=.25
        while gap(reference-half)*gap(reference+half)>0:
            half*=2
            if half>8:raise RuntimeError('No root within 8nm continuation window')
        wavelength=brentq(gap,reference-half,reference+half,xtol=2e-7,rtol=1e-12)
        opt,spec,mode,overlap,residual=evaluate(wavelength)
        if abs(residual)>.05:raise RuntimeError('Frequency closure failed')
        # Same wavelength alone and a nearly parallel field do not identify a
        # nonnormal pole. Compare the TE family, actual C and complex pole too.
        # 中文：同频近共线不等于同模；纵模族、实际算符及复极点身份须一并确认。
        duplicate=closed_duplicate_name(mode,wavelength,opt.result.coupling_m,rows,out,
                                        family=oldrow['te_family'])
        mapping.append(dict(source_name=oldrow['name'],new_name=duplicate or oldrow['name'],duplicate=duplicate is not None,
            reference_overlap=overlap))
        if duplicate:
            print('duplicate',oldrow['name'],duplicate,flush=True);continue
        model=spatial_model(design,opt,mode,wavelength);models.append(model);names.append(mode.name);stored.append((opt,spec,mode))
        row=dict(name=mode.name,te_family=oldrow['te_family'],branch=oldrow['branch'],wavelength_nm=wavelength,
            amplitude_loss_per_m=mode.alpha_per_m,total_power_loss_per_m=model.total_power_loss_per_m,
            up_loss_per_m=model.upward_power_loss_per_m,down_loss_per_m=model.downward_power_loss_per_m,
            gain_weight=float(model.optical_gain_weights.sum()),energy_overlap=float(model.optical_energy_fractions.sum()),
            frequency_residual_per_m=residual,eigen_residual=mode.eigen_residual,reference_overlap=overlap,
            optical_theorem_residual=opt.optical_theorem_residual,ffp=beam(spec,mode),frequency_evaluations=trace)
        rows.append(row);save(out/'optical_modes.json',rows)
        np.savez_compressed(out/(mode.name+'_fields.npz'),field=mode.fields,radiation_x=mode.radiation_x,
            radiation_y=mode.radiation_y,qgain=model.optical_gain_weights,qenergy=model.optical_energy_fractions,
            widths=model.cell_widths_m,lateral=model.lateral_energy_fractions,C=opt.result.coupling_m,
            Lup=opt.upward_amplitude_loss_m,Ldown=opt.downward_amplitude_loss_m)
        print('refined',mode.name,wavelength,mode.alpha_per_m,'overlap',overlap,flush=True)
    system=SpatialMultimodeYbYAG(tuple(models),tuple(names));pumps=[]
    for arch,passes,eta in [('edge_dual',1,1.),('top_single',1,1.),('top_single',20,.98)]:
        pump=PumpGeometryConfig(arch,0.,design.device_um,31,top_passes=passes,return_efficiency=eta)
        threshold=shared_inversion_pump_threshold(system,pump);scans=[]
        for multiplier in (.8,1.05,1.2,1.5,2.,4.):
            power=multiplier*threshold.first_threshold_internal_W
            fit=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
            scans.append(dict(threshold_multiplier=multiplier,input_W=power,active_modes=fit.active_modes,
                outputs_W=dict(zip(names,map(float,fit.upward_output_W))),budget=fit.energy_budget_W,
                ncp_residual=fit.complementarity_residual,population_residual_per_s=fit.population_residual_per_s,
                inactive_gain_excess_per_m=fit.maximum_inactive_gain_excess_per_m))
        pumps.append(dict(architecture=arch,top_passes=passes,return_efficiency=eta,threshold=asdict(threshold),cw_scan=scans,
            metadata=pump_geometry_metadata(models[0],replace(pump,internal_incident_power_W=threshold.first_threshold_internal_W))))
        save(out/'pump_results.json',pumps)
    index=names.index(pumps[0]['threshold']['first_modes'][0])
    plot_vector_far_field_diagnostics(stored[index][1],stored[index][2],out/'selected_vector_ffp.png')
    save(out/'refinement_mapping.json',mapping)
    save(out/'summary.json',dict(design=asdict(design),fixed_period_nm=period,isolated_TE_count=old['isolated_TE_count'],
        included_TE_families=old.get('included_TE_families',[0]),included_modes=names,pump_results=pumps,
        numerics=dict(grid_points=a.grid,CWT_truncation=a.truncation,vertical_step_nm=a.vertical_step_nm),
        scope='frequency-closed continuation of saved lateral poles, not exhaustive finer-mesh pole census',source=str(source)))
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()

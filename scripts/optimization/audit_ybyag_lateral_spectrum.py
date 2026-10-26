"""Challenge a retained Yb:YAG branch with the same finite K/M full spectrum.

This is a fixed-frequency competitor search, not a full frequency-dependent
pole census. Linearized wavelengths guide selection; shortlisted challengers
still require independent C(omega) closure before a modal-selection claim.
"""
import sys,json,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent),str(ROOT/'scripts')]
from optimize_ybyag_cavity import Design,structure,save,spatial_model,beam
from evaluate_ybyag_saved_modes import load_closed
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.custom_analysis import FourWaveOpticalSpec,finite_area_eigenproblem,LinearMode
from pcselsim.frequency_cwt import background_detuning
from pcselsim.ybyag_multimode import _inversion,_arrays
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.run_output import prepare_output,write_provenance,default_output
from run_ybyag_pcsel import lateral_energy_weights


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output')
    p.add_argument('--branch',default='TE0_B');p.add_argument('--truncation',type=int,default=4)
    p.add_argument('--vertical-step-nm',type=float,default=5.)
    a=p.parse_args(argv);source=Path(a.source)
    out=prepare_output(a.output or default_output(ROOT,'optimization_20261021/yag/lateral_challenge'),ROOT)
    write_provenance(out,ROOT);save(out/'run_status.json',dict(status='running'))
    design,system=load_closed(source);summary=json.loads((source/'summary.json').read_text())
    rows=json.loads((source/'optical_modes.json').read_text());row=next(r for r in rows if r['name']==a.branch)
    wavelength=row['wavelength_nm'];seed=np.load(source/(a.branch+'_fields.npz'))['field'];n=seed.shape[-1]
    cell,guide,stack,_=structure(design);guide=replace(guide,te_family=row['te_family'])
    optical=build_reflector_coupling(cell,guide,stack,'passive YAG photonic crystal',summary['fixed_period_nm'],
        wavelength,ThreeDCWTSettings(a.truncation,a.vertical_step_nm),fixed_wavelength_nm=wavelength)
    r=optical.result;spec=FourWaveOpticalSpec(wavelength,r.effective_index,r.group_index,design.device_um,.1,
        tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),n)
    problem=finite_area_eigenproblem(spec,r.coupling_m)
    problem.near(complex(row['frequency_evaluations'][-1]['finite_detuning_per_m'],row['amplitude_loss_per_m']),
                 4,seed.ravel(),dense_limit=2500)
    values,fields=problem._dense_spectrum
    pump=PumpGeometryConfig('edge_dual',0,design.device_um,len(system.modes[0].lateral_energy_fractions))
    threshold=shared_inversion_pump_threshold(system,pump)
    f0,_=_inversion(system,replace(pump,internal_incident_power_W=threshold.first_threshold_internal_W),np.zeros(len(system.modes)))
    volume,weights,constants,_,_=_arrays(system,pump)
    reference_index=system.names.index(a.branch);qgain=system.modes[reference_index].optical_gain_weights
    material=system.modes[reference_index].medium
    gain_z=material.dopant_density_cm3*1e6*np.sum(((material.laser_emission_cross_section_cm2+
        material.laser_absorption_cross_section_cm2)*1e-4*f0-material.laser_absorption_cross_section_cm2*1e-4)*qgain[None,:],axis=1)
    delta_bg=background_detuning(wavelength,r.effective_index,summary['fixed_period_nm'])
    beta=2*np.pi*r.effective_index/(wavelength*1e-9);beta0=2*np.pi/(summary['fixed_period_nm']*1e-9)
    derivative=beta/beta0*(-2*np.pi*r.group_index/(wavelength*1e-9)**2)*1e-9
    approximated_wavelengths=wavelength+(values.real-delta_bg)/derivative
    candidates=[]
    for i,(value,field,estimated) in enumerate(zip(values,fields.T,approximated_wavelengths)):
        if value.imag < -1e-7 or not 1000<=estimated<=1060:continue
        shaped=field.reshape(seed.shape);lateral=lateral_energy_weights(shaped,pump.x_cells)
        gain=float(lateral@gain_z);loss=2*value.imag+100*design.internal_power_loss_cm_inverse
        overlap=float(abs(np.vdot(seed,shaped))**2/(np.sum(abs(seed)**2)*np.sum(abs(shaped)**2)))
        candidates.append(dict(spectrum_index=i,amplitude_loss_per_m=float(value.imag),detuning_per_m=float(value.real),
            estimated_wavelength_nm=float(estimated),gain_to_loss_at_retained_first_threshold=gain/loss,
            overlap_with_retained_seed=overlap))
    candidates.sort(key=lambda x:x['gain_to_loss_at_retained_first_threshold'],reverse=True)
    for candidate in candidates[:24]:
        i=candidate['spectrum_index'];field=fields[:,i].reshape(seed.shape)
        rx,ry=r.radiation_fields(field)
        mode=LinearMode(f'lateral_{i}',values[i].real,values[i].imag,field,0.,rx,ry,discretization='staggered')
        candidate['ffp_at_reference_frequency']=beam(spec,mode)
        np.savez_compressed(out/f'lateral_{i}_seed.npz',field=field,value=values[i])
    save(out/'spectrum_challenge.json',dict(source=str(source),retained_branch=a.branch,
        reference_wavelength_nm=wavelength,grid_points=n,vertical_step_nm=a.vertical_step_nm,
        frequency_scope='fixed C at reference; wavelength estimates linearized, not closed',
        verified_dense_eigenpair_count=len(values),searched_wavelength_estimate_nm=[1000,1060],
        retained_first_threshold_internal_W=threshold.first_threshold_internal_W,candidates=candidates))
    print(json.dumps(candidates[:8],indent=2),flush=True)
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()
